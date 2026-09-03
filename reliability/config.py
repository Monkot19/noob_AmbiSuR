from dataclasses import dataclass


@dataclass(frozen=True)
class CoreConfig:
    """Feature gates for the strictly nested Core ablation ladder."""

    seed: int = 0
    core_shadow_mode: bool = False
    enable_observation_calibration: bool = False
    enable_dual_reliability: bool = False
    enable_abstention: bool = False
    enable_parameter_routing: bool = False
    enable_gradient_projection: bool = False
    enable_reliability_lifecycle: bool = False

    def enabled_features(self):
        features = []
        if self.core_shadow_mode:
            features.append("shadow_diagnostics")
        if self.enable_observation_calibration:
            features.append("observation_calibration")
        if self.enable_dual_reliability:
            features.append("dual_reliability")
        if self.enable_abstention:
            features.append("abstention")
        if self.enable_parameter_routing:
            features.append("parameter_routing")
        if self.enable_gradient_projection:
            features.append("gradient_projection")
        if self.enable_reliability_lifecycle:
            features.append("reliability_lifecycle")
        return tuple(features)

    def uses_core_path(self):
        return bool(self.enabled_features())

    def validate(self):
        requirements = (
            (
                self.enable_dual_reliability,
                self.enable_observation_calibration,
                "dual reliability requires observation calibration",
            ),
            (
                self.enable_abstention,
                self.enable_dual_reliability,
                "abstention requires dual reliability",
            ),
            (
                self.enable_parameter_routing,
                self.enable_abstention,
                "parameter routing requires abstention",
            ),
            (
                self.enable_gradient_projection,
                self.enable_parameter_routing,
                "gradient projection requires parameter routing",
            ),
            (
                self.enable_reliability_lifecycle,
                self.enable_gradient_projection,
                "reliability lifecycle requires gradient projection",
            ),
        )
        for enabled, prerequisite, message in requirements:
            if enabled and not prerequisite:
                raise ValueError(message)
