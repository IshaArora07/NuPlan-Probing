import os
import time
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
import numpy.typing as npt
import shapely
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType

from scipy.special import softmax

from src.feature_builders.nuplan_scenario_render import NuplanScenarioRender
from ..post_processing.emergency_brake import EmergencyBrake
from ..post_processing.trajectory_evaluator import TrajectoryEvaluator
from ..scenario_manager.scenario_manager import ScenarioManager
from .ml_planner_utils import global_trajectory_to_states, load_checkpoint


class PlutoPlanner(AbstractPlanner):
    requires_scenario: bool = True

    def __init__(
        self,
        planner: TorchModuleWrapper,
        scenario: AbstractScenario = None,
        planner_ckpt: str = None,
        render: bool = False,
        use_gpu=True,
        save_dir=None,
        candidate_subsample_ratio: int = 0.5,
        candidate_min_num: int = 1,
        candidate_max_num: int = 20,
        eval_dt: float = 0.1,
        eval_num_frames: int = 80,
        learning_based_score_weight: float = 0.25,
        use_prediction: bool = True,
    ) -> None:

        self._render = render
        self._imgs = []
        self._scenario = scenario

        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )

        self._use_prediction = use_prediction

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt

        self._initialization: Optional[PlannerInitialization] = None
        self._scenario_manager: Optional[ScenarioManager] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1
        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames

        self._candidate_subsample_ratio = candidate_subsample_ratio
        self._candidate_min_num = candidate_min_num
        self._topk = candidate_max_num

        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._trajectory_evaluator = TrajectoryEvaluator(eval_dt, eval_num_frames)
        self._emergency_brake = EmergencyBrake()
        self._learning_based_score_weight = learning_based_score_weight

        if render:
            self._scene_render = NuplanScenarioRender()
            self.video_dir = Path(save_dir if save_dir else os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:

        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval().to(self.device)

        self._initialization = initialization

        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api,
            ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )

        self._planner_feature_builder.scenario_manager = self._scenario_manager

        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:

        start = time.perf_counter()

        ego_state = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()

        trajectory = self._run_planning_once(current_input)

        self._inference_runtimes.append(time.perf_counter() - start)

        return trajectory

    def _run_planning_once(self, current_input: PlannerInput):

        ego_state = current_input.history.ego_states[-1]

        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )

        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor()]
        ).to_device(self.device)

        out = self._planner.forward(planner_feature_torch.data)

        if isinstance(out, tuple):
            out = out[0]

        # ------------------------------------------------
        # Candidate trajectories (EMoE compatible)
        # ------------------------------------------------
        if "candidate_trajectories" in out:

            candidate_trajectories = (
                out["candidate_trajectories"][0].cpu().numpy().astype(np.float64)
            )

        else:
            traj = out["trajectory"]  # [B,1,Ka,T,6]

            traj = traj[0, 0]  # [Ka,T,6]

            candidate_trajectories = (
                traj[..., :3].detach().cpu().numpy().astype(np.float64)
            )

        # ------------------------------------------------
        # Probability
        # ------------------------------------------------
        if "probability" in out:

            prob = out["probability"]

            if prob.ndim == 3:
                probability = prob[0, 0].detach().cpu().numpy()
            else:
                probability = prob[0].detach().cpu().numpy()

        else:
            probability = np.ones(len(candidate_trajectories)) / len(
                candidate_trajectories
            )

        # ------------------------------------------------
        # Prediction key mapping
        # ------------------------------------------------
        pred_key = (
            "output_prediction"
            if "output_prediction" in out
            else ("prediction" if "prediction" in out else None)
        )

        if self._use_prediction and pred_key is not None:
            predictions = out[pred_key][0].detach().cpu().numpy()
        else:
            predictions = None

        # ------------------------------------------------
        # Ref-free trajectory
        # ------------------------------------------------
        ref_free_trajectory = (
            out["output_ref_free_trajectory"][0].cpu().numpy().astype(np.float64)
            if "output_ref_free_trajectory" in out
            else None
        )

        # ------------------------------------------------
        # Candidate trimming
        # ------------------------------------------------
        candidate_trajectories, learning_based_score = self._trim_candidates(
            candidate_trajectories,
            probability,
            current_input.history.ego_states[-1],
            ref_free_trajectory,
        )

        rule_scores = self._trajectory_evaluator.evaluate(
            candidate_trajectories=candidate_trajectories,
            init_ego_state=current_input.history.ego_states[-1],
            detections=current_input.history.observations[-1],
            traffic_light_data=current_input.traffic_light_data,
            agents_info=self._get_agent_info(
                planner_feature.data, predictions, ego_state
            ),
            route_lane_dict=self._scenario_manager.get_route_lane_dicts(),
            drivable_area_map=self._scenario_manager.drivable_area_map,
            baseline_path=self._get_ego_baseline_path(
                self._scenario_manager.get_cached_reference_lines(), ego_state
            ),
        )

        final_scores = rule_scores + self._learning_based_score_weight * learning_based_score

        best_idx = final_scores.argmax()

        trajectory = candidate_trajectories[best_idx, 1:]

        trajectory = InterpolatedTrajectory(
            global_trajectory_to_states(
                global_trajectory=trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(trajectory) * self._step_interval,
                step_interval=self._step_interval,
                include_ego_state=False,
            )
        )

        return trajectory
