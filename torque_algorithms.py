import numpy as np


class TorqueDistributionAlgorithms:
    def __init__(self, controller, env_int, env_float, cp):
        self.ctrl = controller
        self.env_int = env_int
        self.env_float = env_float
        self.cp = cp
        self.debug_fallback = self.env_int("TORQUE_DEBUG_FALLBACK", 0) > 0

    def _yaw_to_axle_delta(self, req_yaw_moment: float, front_share: float) -> tuple[float, float]:
        denom = max(self.ctrl.track_width_m, 1e-6)
        coeff = (self.ctrl.wheel_radius_m) / denom
        front_delta = coeff * req_yaw_moment * np.clip(front_share, 0.0, 1.0)
        rear_delta = coeff * req_yaw_moment * (1.0 - np.clip(front_share, 0.0, 1.0))
        return float(front_delta), float(rear_delta)

    def _base_axle_totals(self, total_torque: float) -> tuple[float, float]:
        front_total = total_torque * self.ctrl.front_ratio
        rear_total = total_torque - front_total
        return float(front_total), float(rear_total)

    def _wheel_loads(self, veh_ax: float, veh_ay: float) -> np.ndarray:
        wheelbase = max(self.ctrl.lf + self.ctrl.lr, 1e-6)
        fz_total = max(self.ctrl.mass_kg * self.ctrl.g, 1e-3)

        fz_front = (self.ctrl.mass_kg * self.ctrl.g * self.ctrl.lr / wheelbase) - (
            self.ctrl.mass_kg * float(veh_ax) * self.ctrl.cg_height_m / wheelbase
        )
        fz_rear = (self.ctrl.mass_kg * self.ctrl.g * self.ctrl.lf / wheelbase) + (
            self.ctrl.mass_kg * float(veh_ax) * self.ctrl.cg_height_m / wheelbase
        )

        fz_front = float(np.clip(fz_front, 0.05 * fz_total, 0.95 * fz_total))
        fz_rear = float(np.clip(fz_rear, 0.05 * fz_total, 0.95 * fz_total))

        track = max(self.ctrl.track_width_m, 1e-6)
        d_fz_total_lat = self.ctrl.mass_kg * float(veh_ay) * self.ctrl.cg_height_m / track
        front_lat_share = fz_front / max(fz_front + fz_rear, 1e-6)
        rear_lat_share = 1.0 - front_lat_share

        d_fz_front_lat = d_fz_total_lat * front_lat_share
        d_fz_rear_lat = d_fz_total_lat * rear_lat_share

        fz_fl = 0.5 * fz_front - 0.5 * d_fz_front_lat
        fz_fr = 0.5 * fz_front + 0.5 * d_fz_front_lat
        fz_rl = 0.5 * fz_rear - 0.5 * d_fz_rear_lat
        fz_rr = 0.5 * fz_rear + 0.5 * d_fz_rear_lat

        fz = np.asarray([fz_fl, fz_fr, fz_rl, fz_rr], dtype=np.float64)
        fz = np.maximum(fz, 0.01 * fz_total)
        return fz

    def _wheel_load_weights(self, veh_ax: float, veh_ay: float) -> np.ndarray:
        fz = self._wheel_loads(veh_ax, veh_ay)
        return fz / max(float(np.sum(fz)), 1e-9)

    def _index_within_bounds(self, value: float, vmin: float, vmax: float) -> float:
        mid = 0.5 * (vmin + vmax)
        half = 0.5 * (vmax - vmin)
        if half <= 1e-9:
            return 1.0
        dist = min(abs(value - vmin), abs(value - vmax))
        inside = 1.0 if (value - vmin) * (vmax - value) >= 0.0 else -1.0
        idx = 1.0 - inside * (dist / max(half, 1e-9))
        return float(np.clip(idx, 0.0, 1.0))

    def _rho_from_indices(self, i_unified: float) -> float:
        if i_unified < self.ctrl.rho_start:
            return 0.0
        if i_unified >= self.ctrl.rho_end:
            return 1.0
        t = (i_unified - self.ctrl.rho_start) / max(self.ctrl.rho_end - self.ctrl.rho_start, 1e-9)
        return float(0.5 * (1.0 - np.cos(np.pi * t)))

    def _estimate_energy_front_share(self, total_torque: float, rotv: np.ndarray, motor_map) -> float:
        omega = np.abs(np.asarray(rotv, dtype=np.float64))
        rpm = omega * 60.0 / (2.0 * np.pi)
        total = float(total_torque)
        if abs(total) < 1e-6:
            return 0.0
        share = motor_map.optimal_front_share(rpm[0], rpm[1], total)
        return float(np.clip(share, 0.0, 1.0))

    def _redistribute_between_groups(self, torques: np.ndarray, src_idx: list[int], dst_idx: list[int], limit: float) -> np.ndarray:
        t = np.asarray(torques, dtype=np.float64).copy()
        src_total = float(np.sum(t[src_idx]))
        src_cap = float(len(src_idx) * limit)
        if src_total > src_cap:
            excess = src_total - src_cap
        elif src_total < -src_cap:
            excess = src_total + src_cap
        else:
            return t

        if excess > 0.0:
            headroom = limit - t[dst_idx]
            headroom = np.maximum(headroom, 0.0)
        else:
            headroom = -limit - t[dst_idx]
            headroom = np.minimum(headroom, 0.0)

        total_room = float(np.sum(np.abs(headroom)))
        if total_room <= 1e-9:
            return np.clip(t, -limit, limit)

        movable = float(np.sign(excess) * min(abs(excess), total_room))
        weights = np.abs(t[src_idx])
        if float(np.sum(weights)) <= 1e-9:
            weights = np.ones(len(src_idx), dtype=np.float64)
        weights = weights / max(float(np.sum(weights)), 1e-9)
        t[src_idx] -= movable * weights
        t[dst_idx] += movable * (headroom / max(float(np.sum(np.abs(headroom))), 1e-9))
        return np.clip(t, -limit, limit)

    def _apply_saturation_redistribution(self, torques: np.ndarray) -> np.ndarray:
        limit = float(self.ctrl.wheel_torque_limit)
        t = np.asarray(torques, dtype=np.float64).copy()

        # Axle transfer: front <-> rear
        t = self._redistribute_between_groups(t, [0, 1], [2, 3], limit)
        t = self._redistribute_between_groups(t, [2, 3], [0, 1], limit)

        # Side transfer: left <-> right
        t = self._redistribute_between_groups(t, [0, 2], [1, 3], limit)
        t = self._redistribute_between_groups(t, [1, 3], [0, 2], limit)

        return np.clip(t, -limit, limit)

    def algo1(self, total_torque: float, req_yaw_moment: float) -> np.ndarray:
        front_total, rear_total = self._base_axle_totals(total_torque)
        front_delta, rear_delta = self._yaw_to_axle_delta(req_yaw_moment, self.ctrl.yaw_front_share)
        fl = 0.5 * front_total - 0.5 * front_delta
        fr = 0.5 * front_total + 0.5 * front_delta
        rl = 0.5 * rear_total - 0.5 * rear_delta
        rr = 0.5 * rear_total + 0.5 * rear_delta
        # print(f"[TORQUE] algo1 raw: fl={fl:.2f}, fr={fr:.2f}, rl={rl:.2f}, rr={rr:.2f}")
        return np.array([fl, fr, rl, rr], dtype=np.float64)


    def algo3(self, total_torque: float, req_yaw_moment: float, veh_speed: float, rotv: np.ndarray, motor_map) -> np.ndarray:
        denom = max(self.ctrl.track_width_m, 1e-6)
        coeff = self.ctrl.wheel_radius_m / denom
        delta_lr = coeff * req_yaw_moment
        left_total = 0.5 * total_torque - 0.5 * delta_lr
        right_total = 0.5 * total_torque + 0.5 * delta_lr

        def _snap_share(share: float) -> float:
            share = float(np.clip(share, 0.0, 1.0))
            if 0.4 <= share <= 0.6:
                return 0.5
            return 1.0

        left_rotv = np.array([rotv[0], rotv[2]], dtype=np.float64)
        # print(self._estimate_energy_front_share(left_total, left_rotv, motor_map))
        # left_share = _snap_share(self._estimate_energy_front_share(left_total, left_rotv, motor_map))
        left_share = self._estimate_energy_front_share(left_total, left_rotv, motor_map)
        if self.env_int("ALGO3_FRONT_SHARE_INVERT", 0) > 0:
            left_share = 1.0 - left_share
        fl = left_total * left_share
        rl = left_total * (1.0 - left_share)

        right_rotv = np.array([rotv[1], rotv[3]], dtype=np.float64)
        # right_share = _snap_share(self._estimate_energy_front_share(right_total, right_rotv, motor_map))
        right_share = self._estimate_energy_front_share(right_total, right_rotv, motor_map)
        if self.env_int("ALGO3_FRONT_SHARE_INVERT", 0) > 0:
            right_share = 1.0 - right_share
        fr = right_total * right_share
        rr = right_total * (1.0 - right_share)
        return np.array([fl, fr, rl, rr], dtype=np.float64)

    def algo2(
        self,
        total_torque: float,
        req_yaw_moment: float,
        veh_speed: float,
        veh_ax: float,
        veh_ay: float,
        yaw_rate: float,
        rotv: np.ndarray,
        motor_map,
        fy_front: float,
        fy_rear: float,
    ) -> np.ndarray:
        fz = self._wheel_loads(veh_ax, veh_ay)
        fz_front = float(fz[0] + fz[1])
        fz_rear = float(fz[2] + fz[3])

        if fz_front <= 1e-6 or fz_rear <= 1e-6:
            return self.algo1(total_torque, req_yaw_moment)

        fy_fl = float(fy_front) * float(fz[0] / max(fz_front, 1e-9))
        fy_fr = float(fy_front) * float(fz[1] / max(fz_front, 1e-9))
        fy_rl = float(fy_rear) * float(fz[2] / max(fz_rear, 1e-9))
        fy_rr = float(fy_rear) * float(fz[3] / max(fz_rear, 1e-9))
        fy = np.array([fy_fl, fy_fr, fy_rl, fy_rr], dtype=np.float64)

        mu = float(self.env_float("TIRE_MU", 0.95))
        wheel_r = max(self.ctrl.wheel_radius_m, 1e-6)
        limit = float(self.ctrl.wheel_torque_limit)
        denom = np.maximum(mu * fz, 1e-3)
        w = (1.0 / (wheel_r * denom)) ** 2

        coeff = (self.ctrl.track_width_m) / max(self.ctrl.wheel_radius_m, 1e-6)
        tau = self.cp.Variable(4)
        mz = coeff * (tau[1] - tau[0] + tau[3] - tau[2])

        fx = tau / wheel_r
        obj = self.cp.sum_squares(self.cp.multiply(np.sqrt(w), tau))
        obj = obj + self.cp.sum_squares(fy / denom)

        constraints = [
            self.cp.sum(tau) == float(total_torque),
            mz == float(req_yaw_moment),
            tau <= limit,
            tau >= -limit,
        ]

        prob = self.cp.Problem(self.cp.Minimize(obj), constraints)
        try:
            prob.solve(solver=self.cp.OSQP, warm_start=True, verbose=False)
            if tau.value is None:
                raise ValueError("algo2 QP failed")
            return np.asarray(tau.value, dtype=np.float64)
        except Exception:
            if self.debug_fallback:
                print("[TORQUE] algo2 QP failed -> fallback to algo1")
            return self.algo1(total_torque, req_yaw_moment)

    def distribute_torque(
        self,
        total_torque: float,
        req_yaw_moment: float,
        veh_speed: float,
        veh_ax: float,
        veh_ay: float,
        yaw_rate: float,
        rotv: np.ndarray,
        motor_map,
        fy_front: float = 0.0,
        fy_rear: float = 0.0,
    ) -> np.ndarray:
        mode = self.ctrl.distribution_mode
        if mode == "algo3":
            torques = self.algo3(total_torque, req_yaw_moment, veh_speed, rotv, motor_map)
        elif mode == "algo2":
            torques = self.algo2(
                total_torque,
                req_yaw_moment,
                veh_speed,
                veh_ax,
                veh_ay,
                yaw_rate,
                rotv,
                motor_map,
                fy_front,
                fy_rear,
            )
        else:
            if mode not in {"algo0", "algo1", "algo2", "algo3"}:
                if self.debug_fallback:
                    print(f"[TORQUE] unknown mode '{mode}' -> fallback to algo1")
            torques = self.algo1(total_torque, req_yaw_moment)
        limit = float(self.ctrl.wheel_torque_limit)
        if mode in {"algo2", "algo3"}:
            if np.any(np.abs(torques) > (limit + 1e-9)):
                # Fallback to vertical-load-based distribution when saturated.
                if self.debug_fallback:
                    print(f"[TORQUE] {mode} saturation -> fallback to algo1 (limit={limit:.2f})")
                torques = self.algo1(total_torque, req_yaw_moment)
        return np.clip(torques, -limit, limit)
