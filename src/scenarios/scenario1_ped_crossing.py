# src/scenarios/scenario1_ped_crossing.py
import time, math, random
import carla

# ------- Scenario Parameters -------
TOWN = "Town03"          # urban map
EGO_SPEED_MS = 20         # ~60 km/h
CROSS_DELAY_S = 1.0      # Δ when pedestrian steps off curb
PED_SPEED_MS = 4
AHEAD_M = 43.0           # ped spawn ahead of ego
LATERAL_M = 10.0         # to the right (curb)
SIM_DT = 0.01            # 100 FPS
RUNTIME_S = 10.0
# -----------------------------------

def set_sync(world, enabled=True, dt=SIM_DT):
    s = world.get_settings()
    s.synchronous_mode = enabled
    s.fixed_delta_seconds = dt if enabled else None
    s.substepping = False
    world.apply_settings(s)

def fwd_vec(rot):
    yaw = math.radians(rot.yaw)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

def move_behind(tf, distance):
    fwd = fwd_vec(tf.rotation)
    new_tf = carla.Transform(tf.location - fwd * distance, tf.rotation)
    return new_tf

def right_vec(rot):
    yaw = math.radians(rot.yaw + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

def choose_straight_spawn(world):
    spawns = world.get_map().get_spawn_points()
    index = 141 
    return spawns[index]

# ---------------------------------------------------------------
#  Tunnel / opposite-side lane placement helpers
# ---------------------------------------------------------------

def yaw_left_unit(rot):
    yaw = math.radians(rot.yaw)
    return math.sin(yaw), -math.cos(yaw)

def transform_on_other_side(world, base_tf, side="left",
                            lanes_guess=3, median_guess=2.0,
                            forward=8.0, up=0.5):
    """
    From a base transform (e.g., spawn point), jump laterally across the divider
    to the *other physical side* (like the other tunnel lane),
    snap to Driving lane, step forward, and lift a bit.
    """
    amap = world.get_map()
    base_wp = amap.get_waypoint(base_tf.location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)

    lane_w = base_wp.lane_width or 3.5
    jump = lanes_guess * lane_w + median_guess
    offset = +jump if side == "left" else -jump

    ahead_wp = base_wp.next(forward)[0]
    ahead_tf = ahead_wp.transform

    lx, ly = yaw_left_unit(ahead_tf.rotation)
    cand = carla.Location(
        x=ahead_tf.location.x + lx * offset,
        y=ahead_tf.location.y + ly * offset,
        z=ahead_tf.location.z
    )

    target_wp = amap.get_waypoint(cand, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
    target_wp = target_wp.next(4.0)[0]
    tf = target_wp.transform
    tf.location.z += up
    return tf

def label(world, loc, text, color=carla.Color(0,255,0), life=10.0):
    world.debug.draw_string(loc, text, False, color, life, True)

# ---------------------------------------------------------------
#  Spectator follow camera
# ---------------------------------------------------------------

def follow_spectator(world, target, dist=7.5, height=2.5):
    spec = world.get_spectator()
    tf = target.get_transform()
    fwd = fwd_vec(tf.rotation)
    cam_loc = tf.location - fwd * dist
    cam_loc.z += height
    spec.set_transform(carla.Transform(cam_loc, tf.rotation))

# ---------------------------------------------------------------
#  Main Scenario
# ---------------------------------------------------------------

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Load map
    world = client.load_world(TOWN)
    time.sleep(0.5)

    bp = world.get_blueprint_library()
    ego_bp = (bp.filter('vehicle.tesla.model3') or bp.filter('vehicle.*'))[0]
    walker_bp = random.choice(bp.filter('walker.pedestrian.*'))

    # Reproducible
    set_sync(world, True, SIM_DT)

    actors = []
    try:
        # Step 1: pick a base spawn
        base_sp = choose_straight_spawn(world)

        # Step 2: jump to the other tunnel side (change side="right" if needed)
        ego_tf = transform_on_other_side(world, base_sp,
                                         side="left",
                                         lanes_guess=3,
                                         median_guess=2.0,
                                         forward=8.0,
                                         up=0.5)
        
        ego_tf = move_behind(ego_tf, 23.0)

        label(world, base_sp.location, "BASE", carla.Color(0,255,0))
        label(world, ego_tf.location, "OTHER SIDE", carla.Color(0,200,255))

        # Step 3: spawn ego
        ego = world.try_spawn_actor(ego_bp, ego_tf)
        if not ego:
            ego_tf = transform_on_other_side(world, base_sp,
                                             side="left",
                                             forward=12.0)
            ego = world.spawn_actor(ego_bp, ego_tf)
        actors.append(ego)
        ego.set_autopilot(False)

        # Step 4: compute pedestrian spawn relative to new ego
        fwd  = fwd_vec(ego_tf.rotation)
        right = right_vec(ego_tf.rotation)
        ped_loc = ego_tf.location + fwd * AHEAD_M + right * LATERAL_M
        ped_tf  = carla.Transform(ped_loc, ego_tf.rotation)
        ped = world.spawn_actor(walker_bp, ped_tf)
        actors.append(ped)

        # Step 5: attach camera
        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "360")
        cam_bp.set_attribute("fov", "90")
        cam = world.spawn_actor(cam_bp,
                                carla.Transform(carla.Location(x=1.5, z=2.2)),
                                attach_to=ego)
        actors.append(cam)
        cam.listen(lambda _: None)

        world.debug.draw_string(ped_loc, "PED START", False,
                                carla.Color(255,0,0), 10.0, True)

        # Simulation loop
        ticks_to_delay = int(CROSS_DELAY_S / SIM_DT)
        ticks_total = int(RUNTIME_S / SIM_DT)
        print("[Scenario] Starting. Pedestrian crosses after delay.")

        for tick in range(ticks_total):
            world.tick()
            follow_spectator(world, ego)

            # Maintain target speed
            v = ego.get_velocity()
            speed = (v.x*v.x + v.y*v.y)**0.5
            throttle = 1 if speed < EGO_SPEED_MS else 0.0
            brake = 0.2 if speed > EGO_SPEED_MS + 0.5 else 0.0
            ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0))

            # Trigger pedestrian crossing
            if tick == ticks_to_delay:
                dir_vec = carla.Vector3D(-right.x, -right.y, -right.z)
                ped_ctrl = carla.WalkerControl(direction=dir_vec, speed=PED_SPEED_MS)
                ped.apply_control(ped_ctrl)
                print("[Scenario] Pedestrian started crossing.")

            # Stop pedestrian after ~20 m
            if tick > ticks_to_delay:
                rel = ped.get_location() - ped_loc
                proj = rel.x * (-right.x) + rel.y * (-right.y)
                if proj > 20.0:
                    ped.apply_control(carla.WalkerControl(speed=0.0))

        print("[Scenario] Finished.")

    finally:
        for a in actors[::-1]:
            try:
                if hasattr(a, "stop"):
                    a.stop()
            except Exception:
                pass
            try:
                a.destroy()
            except Exception:
                pass
        set_sync(world, False)
        print("[Scenario] Clean shutdown & sync disabled.")

if __name__ == "__main__":
    main()