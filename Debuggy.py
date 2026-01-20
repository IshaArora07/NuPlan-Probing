        try:
            feat = self._build_feature(
                present_idx=self.history_samples,
                ego_state_list=ego_state_list,
                tracked_objects_list=tracked_objects_list,
                route_roadblocks_ids=scenario.get_route_roadblock_ids(),
                map_api=scenario.map_api,
                mission_goal=scenario.get_mission_goal(),
                traffic_light_status=scenario.get_traffic_light_status_at_iteration(iteration),
                inference=False,
                scenario_token=scenario_token,
            )
        except Exception as e:
            raise RuntimeError(
                f"[PlutoFeatureBuilder] _build_feature threw. token={scenario_token} "
                f"log={getattr(scenario, 'log_name', None)}"
            ) from e

        if feat is None:
            raise RuntimeError(
                f"[PlutoFeatureBuilder] _build_feature returned None. token={scenario_token}"
            )

        # Hard validity checks
        if not isinstance(feat, PlutoFeature):
            raise RuntimeError(
                f"[PlutoFeatureBuilder] Expected PlutoFeature, got {type(feat)} token={scenario_token}"
            )

        # Print and then crash if invalid (so nuPlan can't return None silently)
        if not feat.is_valid:
            ref_present = "reference_line" in feat.data
            ref_any = None
            ref_shape = None
            if ref_present:
                vm = feat.data["reference_line"]["valid_mask"]
                ref_any = bool(vm.any())
                ref_shape = getattr(vm, "shape", None)

            map_polys = feat.data["map"]["point_position"].shape[0] if "map" in feat.data else None
            raise RuntimeError(
                "[PlutoFeatureBuilder] Produced INVALID PlutoFeature -> would cause caching to drop sample.\n"
                f"token={scenario_token}\n"
                f"ref_present={ref_present}, ref_any={ref_any}, ref_shape={ref_shape}\n"
                f"map_polygons={map_polys}\n"
                f"route_roadblock_ids_len={len(scenario.get_route_roadblock_ids())}\n"
            )

        return feat
