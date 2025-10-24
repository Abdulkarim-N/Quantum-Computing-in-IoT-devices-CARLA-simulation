import time, math, random
import carla

# ------- Parameters -------
TOWN = "Town04"
EGO_SPEED_MS = 22.2
CUTIN_DELAY_S = 3.0
NPC_SPEED_MS = 20.0
SIM_DT = 0.01
RUNTIME_S = 20.0
# ---------------------------

def set_sync(world, enabled=True, dt=SIM_DT):
    s = world.get_settings()
    s.synchronous_mode = enabled
    s.fixed_delta_seconds = dt if enabled else None
    s.substepping = False
    world.apply_settings(s)

def fwd(rot):
    yaw = math.radians(rot.yaw)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0)

def right(rot):
    yaw = math.radians(rot.yaw + 90)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0)

def choose_highway_spawn(world):
    spawns = world.get_map().get_spawn_points()
    # index ~40–60 are highway segments in Town04
    return spawns[42]

def follow_spectator(world, target, dist=10, height=3):
    spec = world.get_spectator()
    tf = target.get_transform()
    fv = fwd(tf.rotation)
    loc = tf.location - fv * dist
    loc.z += height
    spec.set_transform(carla.Transform(loc, tf.rotation))

def main():
    client = carla.Client("localhost", 2000); client.set_timeout(10)
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != TOWN:
        world = client.load_world(TOWN); time.sleep(0.5)
    bp = world.get_blueprint_library()

    ego_bp = (bp.filter('vehicle.tesla.model3') or bp.filter('vehicle.*'))[0]
    npc_bp = bp.filter('vehicle.audi.tt')[0]
    set_sync(world, True)

    actors = []
    try:
        ego_spawn = choose_highway_spawn(world)
        ego = world.spawn_actor(ego_bp, ego_spawn); actors.append(ego)
        ego.set_autopilot(False)

        fvec, rvec = fwd(ego_spawn.rotation), right(ego_spawn.rotation)
        # spawn NPC ahead but in opposing lane
        npc_loc = ego_spawn.location + fvec * 60 - rvec * 3.5
        npc_tf = carla.Transform(npc_loc, ego_spawn.rotation)
        npc_tf.rotation.yaw += 180  # face ego
        npc = world.spawn_actor(npc_bp, npc_tf); actors.append(npc)
        npc.set_autopilot(False)

        # Attach camera (optional)
        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x","640")
        cam_bp.set_attribute("image_size_y","360")
        cam_bp.set_attribute("fov","90")
        cam = world.spawn_actor(cam_bp,
            carla.Transform(carla.Location(x=1.5,z=2.2)), attach_to=ego)
        actors.append(cam); cam.listen(lambda _: None)

        ticks_total = int(RUNTIME_S / SIM_DT)
        cutin_tick = int(CUTIN_DELAY_S / SIM_DT)
        print("[Scenario 2] Starting – NPC will cut in after Δ seconds.")

        for tick in range(ticks_total):
            world.tick()
            follow_spectator(world, ego)

            # Ego cruise control
            v = ego.get_velocity(); speed = math.hypot(v.x, v.y)
            ego.apply_control(carla.VehicleControl(
                throttle = 0.6 if speed < EGO_SPEED_MS else 0.0,
                brake    = 0.3 if speed > EGO_SPEED_MS + 0.5 else 0.0,
                steer    = 0.0
            ))

            # NPC initial motion (steady opposite lane)
            if tick < cutin_tick:
                npc.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

            # Trigger cut-in
            if tick == cutin_tick:
                print("[Scenario 2] NPC initiating lane cut.")
                npc.apply_control(carla.VehicleControl(throttle=0.5, steer=0.4))  # steer right into ego lane

            # Continue cut-in for a bit, then straighten
            if cutin_tick < tick < cutin_tick + 60:
                npc.apply_control(carla.VehicleControl(throttle=0.4, steer=0.4))
            elif tick == cutin_tick + 60:
                npc.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))

        print("[Scenario 2] Finished.")

    finally:
        for a in actors[::-1]:
            try:
                if hasattr(a,"stop"): a.stop()
            except: pass
            try: a.destroy()
            except: pass
        set_sync(world, False)
        print("[Scenario 2] Clean shutdown and sync disabled.")

if __name__ == "__main__":
    main()
