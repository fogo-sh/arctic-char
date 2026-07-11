package game

import "core:math/linalg"
import protocol "../protocol"
import b3 "vendor:box3d"

REPLICATED_TRANSFORM_SAMPLE_CAPACITY :: 8

ReplicatedKind :: enum {
	None,
	Suzanne,
}

ReplicatedPropAuthority :: enum {
	ServerAuthoritative,
	ClientOnly,
}

ReplicatedTransformSample :: struct {
	server_tick: u32,
	position:    Vec3,
	rotation:    linalg.Quaternionf32,
}

ReplicatedTransformBuffer :: struct {
	count:   int,
	samples: [REPLICATED_TRANSFORM_SAMPLE_CAPACITY]ReplicatedTransformSample,
}

ReplicatedObject :: struct {
	net_id: protocol.NetId,
	kind:   ReplicatedKind,
	authority: ReplicatedPropAuthority,
	transform_buffer: ReplicatedTransformBuffer,
	last_replicated_tick: u32,
}

replicated_transform_add_sample :: proc(buffer: ^ReplicatedTransformBuffer, sample: ReplicatedTransformSample) {
	for i in 0..<buffer.count {
		if buffer.samples[i].server_tick == sample.server_tick {
			buffer.samples[i] = sample
			return
		}
	}

	if buffer.count >= REPLICATED_TRANSFORM_SAMPLE_CAPACITY {
		copy(buffer.samples[:], buffer.samples[1:])
		buffer.count -= 1
	}

	insert_at := buffer.count
	for insert_at > 0 && buffer.samples[insert_at - 1].server_tick > sample.server_tick {
		buffer.samples[insert_at] = buffer.samples[insert_at - 1]
		insert_at -= 1
	}
	buffer.samples[insert_at] = sample
	buffer.count += 1
}

replicated_transform_at_tick :: proc(buffer: ^ReplicatedTransformBuffer, fallback_position: Vec3, fallback_rotation: linalg.Quaternionf32, render_tick: u32) -> (position: Vec3, rotation: linalg.Quaternionf32) {
	if buffer.count == 0 {
		return fallback_position, fallback_rotation
	}
	if buffer.count == 1 || render_tick <= buffer.samples[0].server_tick {
		sample := buffer.samples[0]
		return sample.position, sample.rotation
	}

	last_index := buffer.count - 1
	if render_tick >= buffer.samples[last_index].server_tick {
		sample := buffer.samples[last_index]
		return sample.position, sample.rotation
	}

	for i in 0..<last_index {
		from := buffer.samples[i]
		to := buffer.samples[i + 1]
		if render_tick >= from.server_tick && render_tick <= to.server_tick {
			span := to.server_tick - from.server_tick
			if span == 0 {
				return to.position, to.rotation
			}
			t := f32(render_tick - from.server_tick) / f32(span)
			return scene_lerp_vec3(from.position, to.position, t), linalg.quaternion_slerp(from.rotation, to.rotation, t)
		}
	}

	sample := buffer.samples[last_index]
	return sample.position, sample.rotation
}

scene_upsert_replicated_suzanne :: proc(scene: ^Scene, net_id: protocol.NetId, position: Vec3, rotation: linalg.Quaternionf32, server_tick: u32) {
	if object := scene_object_by_net_id(scene, net_id); object != nil {
		if b3.IS_NON_NULL(object.physics.body) {
			b3.DestroyBody(object.physics.body)
		}
		object.kind = .Suzanne
		object.transform.position = position
		object.render_rotation = rotation
		object.render.mesh = scene.suzanne_mesh
		object.render.visible = true
		object.physics = {enabled = false}
		object.replica.kind = .Suzanne
		object.replica.authority = .ServerAuthoritative
		object.replica.last_replicated_tick = server_tick
		replicated_transform_add_sample(&object.replica.transform_buffer, {server_tick = server_tick, position = position, rotation = rotation})
		return
	}

	if len(scene.objects) >= cap(scene.objects) {
		return
	}
	object := Object{
		name = "Suzanne",
		kind = .Suzanne,
		transform = {position = position},
		render_rotation = rotation,
		render = {mesh = scene.suzanne_mesh, visible = true},
		physics = {enabled = false},
		replica = {
			net_id = net_id,
			kind = .Suzanne,
			authority = .ServerAuthoritative,
			last_replicated_tick = server_tick,
		},
	}
	replicated_transform_add_sample(&object.replica.transform_buffer, {server_tick = server_tick, position = position, rotation = rotation})
	scene_add_object(scene, object)
}

scene_remove_replicated_prop :: proc(scene: ^Scene, net_id: protocol.NetId) {
	for &object, index in scene.objects {
		if object.replica.net_id != net_id || object.replica.kind == .None {
			continue
		}
		if object.physics.enabled && b3.IS_NON_NULL(object.physics.body) {
			b3.DestroyBody(object.physics.body)
		}
		ordered_remove(&scene.objects, index)
		return
	}
}

scene_object_by_net_id :: proc(scene: ^Scene, net_id: protocol.NetId) -> ^Object {
	for &object in scene.objects {
		if object.replica.net_id == net_id && object.replica.kind != .None {
			return &object
		}
	}
	return nil
}
