"""Simon-Says expression definitions for rig calibration.

The expressions here define the set of ``(name, controls_dict)`` pairs
used in the Simon-Says calibration process.  Each expression specifies
which semantic controls should be active and at what value.

Users can define their own expression sets for custom rigs.
"""

import numpy as np

# ---------------------------------------------------------------------------
# FLAME semantic control expressions (20 expressions)
# ---------------------------------------------------------------------------

FLAME_SIMON_SAYS_20 = [
    ("Neutral", {}),
    ("Brows Down", {
        "browDownL": 1,
        "browDownR": 1,
    }),
    ("Brows Raise", {
        "browRaiseInL": 1,
        "browRaiseInR": 1,
        "browRaiseOuterL": 1,
        "browRaiseOuterR": 1,
    }),
    ("Eyes Wide", {
        "eyeWidenL": 1,
        "eyeWidenR": 1,
    }),
    ("Eyes Close", {
        "eyeBlinkL": 1,
        "eyeBlinkR": 1,
    }),
    ("Nose Scrunch/Wrinkle", {
        "noseWrinkleL": 1,
        "noseWrinkleR": 1,
        "noseNasolabialDeepenL": 1,
        "noseNasolabialDeepenR": 1,
    }),
    ("Cheek Puff", {
        "mouthCheekBlowL": 1,
        "mouthCheekBlowR": 1,
        "mouthLipsBlowL": 1,
        "mouthLipsBlowR": 1,
    }),
    ("Teeth Grimace", {
        "mouthUpperLipRaiseL": 1,
        "mouthUpperLipRaiseR": 1,
        "mouthLowerLipDepressL": 1,
        "mouthLowerLipDepressR": 1,
    }),
    ("Smile (corner pull)", {
        "mouthCornerPullL": 1,
        "mouthCornerPullR": 1,
    }),
    ("Mouth Stretch", {
        "mouthStretchL": 1,
        "mouthStretchR": 1,
    }),
    ("Mouth Corner Depress", {
        "mouthCornerDepressL": 1,
        "mouthCornerDepressR": 1,
    }),
    ("Inner Lip Press", {
        "mouthPressUL": 1,
        "mouthPressUR": 1,
        "mouthPressDL": 1,
        "mouthPressDR": 1,
    }),
    ("Pursed Lips", {
        "mouthLipsPurseUL": 1,
        "mouthLipsPurseUR": 1,
        "mouthLipsPurseDL": 1,
        "mouthLipsPurseDR": 1,
    }),
    ("Kissy Lips", {
        "mouthLipsPurseUL": 1,
        "mouthLipsPurseUR": 1,
        "mouthLipsPurseDL": 1,
        "mouthLipsPurseDR": 1,
        "mouthLipsTowardsUL": 1,
        "mouthLipsTowardsUR": 1,
        "mouthLipsTowardsDL": 1,
        "mouthLipsTowardsDR": 1,
    }),
    ("Mouth Funnel", {
        "mouthFunnelUL": 1,
        "mouthFunnelUR": 1,
        "mouthFunnelDL": 1,
        "mouthFunnelDR": 1,
    }),
    ("Lip Bite", {
        "mouthUpperLipBiteL": 1,
        "mouthUpperLipBiteR": 1,
        "mouthLowerLipBiteL": 1,
        "mouthLowerLipBiteR": 1,
    }),
    ("Jaw Open", {"jawOpen": 1}),
    ("Jaw Open Extreme", {
        "jawOpen": 1,
        "jawOpenExtreme": 1,
    }),
    ("Jaw Left", {"jawLeft": 1}),
    ("Jaw Right", {"jawRight": 1}),
]

# Regularization combination expressions
FLAME_COMBINATIONS = [
    ("COMBO Jaw Open Lips Together", {
        "jawOpen": 1,
        "mouthLipsTogetherUL": 1,
        "mouthLipsTogetherUR": 1,
        "mouthLipsTogetherDL": 1,
        "mouthLipsTogetherDR": 1,
    }),
    ("COMBO Jaw Open Lips Purse", {
        "jawOpen": 1,
        "mouthLipsPurseUL": 1,
        "mouthLipsPurseUR": 1,
        "mouthLipsPurseDL": 1,
        "mouthLipsPurseDR": 1,
    }),
    ("COMBO B Pop Phoneme", {
        "jawOpen": 0.25,
        "mouthUpperLipBiteL": 0.25,
        "mouthUpperLipBiteR": 0.25,
        "mouthLowerLipBiteL": 0.25,
        "mouthLowerLipBiteR": 0.25,
        "mouthLipsTogetherUL": 0.25,
        "mouthLipsTogetherUR": 0.25,
        "mouthLipsPurseUL": 0.5,
        "mouthLipsPurseUR": 0.5,
        "mouthLipsPurseDL": 0.5,
        "mouthLipsPurseDR": 0.5,
        "mouthStretchL": 0.5,
        "mouthStretchR": 0.5,
    }),
]


def create_expressions(neutral_controls, control_names, expressions=None):
    """Build control arrays from expression definitions.

    Args:
        neutral_controls: (num_controls,) default control values (usually zeros).
        control_names: list of semantic control names matching the rig.
        expressions: list of ``(name, {ctrl_name: value})`` pairs.
            Defaults to :data:`FLAME_SIMON_SAYS_20`.

    Returns:
        Tuple of ``(expression_names, controls_array, active_control_names)``:

        - expression_names: list of str
        - controls_array: list of (num_controls,) arrays
        - active_control_names: list of lists of active control names per expression
    """
    if expressions is None:
        expressions = FLAME_SIMON_SAYS_20

    controls_map = {name: i for i, name in enumerate(control_names)}
    expression_names, controls_out, active_controls = [], [], []

    for name, ctrl_dict in expressions:
        expression_names.append(name)
        controls = neutral_controls.copy()
        active = []
        for ctrl_name, value in ctrl_dict.items():
            if ctrl_name not in controls_map:
                raise KeyError(
                    f"Control '{ctrl_name}' not found in control_names. "
                    f"Check your expression definitions match the rig."
                )
            controls[controls_map[ctrl_name]] = value
            active.append(ctrl_name)
        controls_out.append(controls)
        active_controls.append(active)

    return expression_names, controls_out, active_controls


def create_flame_expressions(neutral_controls, control_names):
    """Create the 20 FLAME Simon-Says expressions."""
    return create_expressions(neutral_controls, control_names, FLAME_SIMON_SAYS_20)


def create_flame_combinations(neutral_controls, control_names):
    """Create FLAME combination/regularization expressions."""
    return create_expressions(neutral_controls, control_names, FLAME_COMBINATIONS)
