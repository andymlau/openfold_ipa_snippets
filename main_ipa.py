import torch
import torch.nn as nn

from nndef_ipa import InvariantPointAttention
from nndef_ipa_primitives import Rotation, Rigid, LayerNorm, Linear

DEVICE = torch.device("cuda")


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
        no_blocks: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        dropout_rate: float,
        no_transition_layers: int,
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_transition_layers = no_transition_layers

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

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        rigids: Rigid = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Standalone IPA module from openfold/AF2. Updates single representation
        tensor with pair/rigid information.

        If rigids not provided, will initialise frames, a la AlphaFold2.
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

        return s


def main():
    """ Example usage of the IPA module, returning updated single_repr only.

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
    """

    # Set default sizes for the representations and network - all params from
    # openfold/AlphaFold2

    c_s, c_z = 384, 128

    network = IPA(
        c_s=c_s,
        c_z=c_z,
        c_hidden=16,
        no_blocks=8,
        no_heads=12,
        no_qk_points=4,
        no_v_points=8,
        dropout_rate=0.1,
        no_transition_layers=1,
    ).to(DEVICE)

    # Example 200 residue protein
    batches = 1
    residues = 200

    s = torch.rand(batches, residues, c_s).to(DEVICE)
    z = torch.rand(batches, residues, residues, c_z).to(DEVICE)

    # Rigids are optional in this implementation - if not provided, new empty
    # frames will be set up via "Black hole initialisation"
    rigids = Rigid.identity(
        s.shape[:-1],
        s.dtype,
        s.device,
        requires_grad=True,
        fmt="quat",
    )

    out = network(s, z, rigids)

    print("\nInput tensors:")
    print("     Single representation:", s.shape)
    print("     Pair representation:", z.shape)

    print("\nOutput tensors:")
    print("     Updated single rep:", out.shape)


if __name__ == "__main__":
    main()
