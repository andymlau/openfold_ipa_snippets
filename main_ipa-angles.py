from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn

from nndef_ipa import InvariantPointAttention
from nndef_ipa_feats import torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos
from nndef_ipa_primitives import Rotation, Rigid, LayerNorm, Linear, dict_multimap

# Constants from openfold/np/residue_constants
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=np.int)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)

DEVICE = torch.device("cuda")


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")
        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a
        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class IPA(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        c_resnet: int,
        no_blocks: int,
        no_heads: int,
        no_angles: int,
        no_resnet_blocks: int,
        no_qk_points: int,
        no_v_points: int,
        dropout_rate: float,
        no_transition_layers: int,
        trans_scale_factor: int,
        inf: float = 1e8,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.inf = inf
        self.epsilon = eps

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.c_resnet = c_resnet
        self.no_blocks = no_blocks
        self.no_heads = no_heads
        self.no_angles = no_angles
        self.no_resnet_blocks = no_resnet_blocks
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_transition_layers = no_transition_layers
        self.trans_scale_factor = trans_scale_factor

        # To be lazily initialized later
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            c_s=self.c_s,
            c_z=self.c_z,
            c_hidden=self.c_hidden,
            no_heads=self.no_heads,
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):
        # [*, N, 8]  # [*, N]

        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        rigids: Rigid = None,
        mask: torch.Tensor = None
    ) -> dict:
        """
        Standalone IPA module from openfold/AF2. Updates single
        representation tensor with pair/rigid information, and predicts
        angles, frames and coordinates.
        """

        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)

        s_initial = s
        s = self.linear_in(s)

        if rigids is None:
            rigids = Rigid.identity(
                s.shape[:-1],
                s.dtype,
                s.device,
                requires_grad=True,
                fmt="quat",
            )

        if mask is None:
            mask = s.new_ones(s.shape[:-1])  # [*, N]

        outputs = []
        for i in range(self.no_blocks):
            s = s + self.ipa(s, z, rigids, mask)  # [*, N, C_s]
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            backb_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(),
                    quats=None
                ),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                self.trans_scale_factor
            )

            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            # Algorithm 24: compute all atom coordinates
            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }

            outputs.append(preds)

            if i < (self.no_blocks - 1):
                rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs


def main():
    """ This example uses a single and pair tensor to predict frames, angles and
        coordinates using OpenFold's implementation of the IPA module.

        Inputs:
        ------------------------------------------------------------------
            s       [*, n, c_s]     Where c_s is the single_repr hidden dim.
            z       [*, n, n, c_z]  Where c_z is the pair_repr hidden dim.
            aatype  [*, n]          Int-encoded aa types. Only needed for
                                    post-IPA frames/angle predictions.

        Optional inputs:
        ------------------------------------------------------------------
            r       [*, n, 3, 3]    Rotation matrix from backbone frames.
            t       [*, n, 3]       Translation matrix for backbone frames.
                                    Are also the C-alpha coordinates.

                                    If r, t are not provided to forward(),
                                    network will use "Black Hole initialisation".

            If providing your own r and t, they need to be wrapped in custom
            'Rigid' and 'Rotation' classes. This can be done as:

            rigids = Rigid(Rotation(r), t)

        Outputs:
        ------------------------------------------------------------------
            dictionary with keys:

            single              [*, n, c_s]     Updated single representation.
                                                Same dimension as original input.

            angles              [no_blocks, *, n, 7, 2]
            positions           [no_blocks, *, n, 14, 3]
            frames              [no_blocks, *, n, 7]
            sidechain_frames    [no_blocks, *, n, 8, 4, 4]
            unnormalized_angles [no_blocks, *, n, 7, 2]

            Angles are predicted as phi, psi, omega, chi1-4 angles.
            Positions are xyz coordinates for 14 atom types.

    """

    # Set default sizes for the representations and network - all params from
    # openfold/AlphaFold2

    c_s, c_z, c_resnet = 384, 128, 128

    network = IPA(
        c_s=c_s,
        c_z=c_z,
        c_hidden=16,
        c_resnet=c_resnet,
        no_blocks=8,
        no_heads=12,
        no_angles=7,
        no_resnet_blocks=2,
        no_qk_points=4,
        no_v_points=8,
        dropout_rate=0.1,
        no_transition_layers=1,
        trans_scale_factor=10,
    ).to(DEVICE)

    # Example input
    b, n = 1, 200   # batch_size, n_res

    s = torch.rand(b, n, c_s).to(DEVICE)
    z = torch.rand(b, n, n, c_z).to(DEVICE)
    aatype = torch.randint(0, 21, (b, n)).to(DEVICE)

    outputs = network(s, z, aatype)

    print("\nInput tensors:")
    print("     Single representation:", s.shape)
    print("     Pair representation:", z.shape)

    print("\nOutput tensors:")
    print("     Updated single rep:", outputs['single'].shape)

    print("\n     frames tensor:", outputs['frames'].shape)
    print("     sidechain_frames tensor:", outputs['sidechain_frames'].shape)
    print("     unnormalized_angles tensor:", outputs['unnormalized_angles'].shape)
    print("     angles tensor:", outputs['angles'].shape)
    print("     positions tensor:", outputs['positions'].shape)

    # Input tensors:
    #      Single representation: torch.Size([1, 200, 384])
    #      Pair representation: torch.Size([1, 200, 200, 128])
    #
    # Output tensors:
    #      Updated single rep: torch.Size([1, 200, 384])
    #
    #      frames tensor: torch.Size([8, 1, 200, 7])
    #      sidechain_frames tensor: torch.Size([8, 1, 200, 8, 4, 4])
    #      unnormalized_angles tensor: torch.Size([8, 1, 200, 7, 2])
    #      angles tensor: torch.Size([8, 1, 200, 7, 2])
    #      positions tensor: torch.Size([8, 1, 200, 14, 3])


if __name__ == "__main__":
    main()
