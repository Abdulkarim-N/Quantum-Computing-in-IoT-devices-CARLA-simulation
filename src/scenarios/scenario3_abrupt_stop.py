import time, math, random
import carla

# ======= Scenario 3: Abrupt Stop of Lead Vehicle =======
# Ego and lead vehicle travel same lane/direction. After Δ seconds,
# the lead applies strong braking to a standstill. Ego must brake or evade.
# ========================================================

# --- Parameters you can tweak ---
TOWN = "Town03"             # two-way urban, easy to visualize
SIM_DT = 0.01               # 100 FPS synchronous
RUNTIME_S = 20.0
EGO_SPEED_MS = 16.7         # ~60 km/h cruise
LEAD_INIT_GAP_M = 45.0      # distance ahead of ego to place lead car
DELTA_TRIG_S = 3.0          # time when lead begins abrupt braking
LEAD_BRAKE_SECS = 2.0       # how long we command heavy braking
EGO_CRUISE_THROTTLE = 0.6   # simple cruise control
# -------------------------------

def set_sync(world, enabled=True, dt=SIM_DT):
    s = world.get_settings()
    s.synchronous_mode = enabled
    s.fixed_delta_seconds = dt if enabled else None
    s.substepping = False
    world.apply_settings(s)

def fwd(rot):
    yaw = math.radians(rot.yaw)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

def right(rot):
    yaw = math.radians(rot.yaw + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

def choose_straight_spawn(world):
    # A deterministic, straightish segment. Adjust index if your Town03 differs.
    spawns = world.get_map().get_spawn_points()
    return spawns[12]

def follow_spectator(world, target, dist=9.0, height=3.0):
    spec = world.get_spectator()
    tf = target.get_transform()
    fv = fwd(tf.rotation)
    loc = tf.location - fv * dist
    loc.z += height
    spec.set_transform(carla.Transform(loc, tf.rotation))

def speed_ms(actor):
    v = actor.get_velocity()
    return math.hypot(v.x, v.y)

def main():
    client = carla.Client("localhost", 2000); client.set_timeout(10.0)
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != TOWN:
        world = client.load_world(TOWN); time.sleep(0.5)

    bp = world.get_blueprint_library()
    ego_bp  = (bp.filter('vehicle.tesla.model3') or bp.filter('vehicle.*'))[0]
    lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]

    set_sync(world, True, SIM_DT)
    actors = []
    try:
        # Spawn ego
        ego_sp = choose_straight_spawn(world)
        ego = world.spawn_actor(ego_bp, ego_sp); actors.append(ego)
        ego.set_autopilot(False)

        fvec = fwd(ego_sp.rotation)

        # Spawn lead ahead in same lane & orientation
        lead_loc = carla.Location(
            x = ego_sp.location.x + fvec.x * LEAD_INIT_GAP_M,
            y = ego_sp.location.y + fvec.y * LEAD_INIT_GAP_M,
            z = ego_sp.location.z
        )
        lead_tf  = carla.Transform(lead_loc, ego_sp.rotation)
        lead = world.spawn_actor(lead_bp, lead_tf); actors.append(lead)
        lead.set_autopilot(False)

        # Optional: attach a small RGB camera (harmless if sensors.py also runs)
        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x","640")
        cam_bp.set_attribute("image_size_y","360")
        cam_bp.set_attribute("fov","90")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5,z=2.2)), attach_to=ego)
        actors.append(cam); cam.listen(lambda _: None)

        # Debug markers
        world.debug.draw_string(
            lead_loc,
            "LEAD START",
            draw_shadow=False,
            color=carla.Color(255, 140, 0),
            life_time=10.0,
            persistent_lines=False
        )

        total_ticks = int(RUNTIME_S / SIM_DT)
        trig_tick   = int(DELTA_TRIG_S / SIM_DT)
        brake_ticks = int(LEAD_BRAKE_SECS / SIM_DT)

        print("[Scenario 3] Abrupt Stop: Lead ahead will hard-brake at Δ.")
        for tick in range(total_ticks):
            world.tick()
            follow_spectator(world, ego)

            # Ego simple cruise near target speed
            s_ego = speed_ms(ego)
            ego.apply_control(carla.VehicleControl(
                throttle = EGO_CRUISE_THROTTLE if s_ego < EGO_SPEED_MS else 0.0,
                brake    = 0.3 if s_ego > EGO_SPEED_MS + 0.5 else 0.0,
                steer    = 0.0
            ))

            # Lead initial motion: keep near ego speed (slightly slower)
            if tick < trig_tick:
                s_lead = speed_ms(lead)
                target = EGO_SPEED_MS * 0.95
                lead.apply_control(carla.VehicleControl(
                    throttle = 0.5 if s_lead < target else 0.0,
                    brake    = 0.3 if s_lead > target + 0.5 else 0.0,
                    steer    = 0.0
                ))

            # At Δ: abrupt braking phase
            if trig_tick <= tick < trig_tick + brake_ticks:
                if tick == trig_tick:
                    print("[Scenario 3] Lead begins hard braking.")
                lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.9, hand_brake=False))

            # After braking window, keep it stopped
            if tick >= trig_tick + brake_ticks:
                lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

            # (Optional) Stop early if ego gets within 5 m of lead center (for safety)
            gap = (lead.get_location() - ego.get_location())
            gap_xy = math.hypot(gap.x, gap.y)
            if gap_xy < 5.0:
                print("[Scenario 3] Early stop: ego is within 5 m of lead.")
                break

        print("[Scenario 3] Finished.")

    finally:
        for a in actors[::-1]:
            try:
                if hasattr(a, "stop"): a.stop()
            except Exception: pass
            try: a.destroy()
            except Exception: pass
        set_sync(world, False)
        print("[Scenario 3] Clean shutdown; sync disabled.")

if __name__ == "__main__":
    main()
