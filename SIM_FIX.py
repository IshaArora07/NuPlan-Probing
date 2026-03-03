def patch_emoe_outputs():
    """
    Patch planner outputs so that simulation/video callbacks
    receive candidate_trajectories even if the model does not
    return them.
    """

    import torch
    from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

    original_compute = AbstractPlanner.compute_trajectory

    def wrapped_compute(self, current_input):
        output = original_compute(self, current_input)

        # If already correct, do nothing
        if isinstance(output, dict) and "candidate_trajectories" in output:
            return output

        # Handle EMoE model output
        if isinstance(output, dict) and "trajectory" in output:

            traj = output["trajectory"]

            if isinstance(traj, torch.Tensor) and traj.ndim == 5:
                # [B,1,Ka,T,6]
                traj = traj[0, 0]

                candidates = traj[..., :3]  # [Ka,T,3]

                if "probability" in output:
                    probs = output["probability"][0, 0]
                    best = int(torch.argmax(probs))
                else:
                    best = 0

                best_traj = candidates[best]

                output["candidate_trajectories"] = candidates
                output["trajectory"] = best_traj

        return output

    AbstractPlanner.compute_trajectory = wrapped_compute
