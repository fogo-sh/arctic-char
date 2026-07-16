package engine

RendererApi :: struct {
	data: rawptr,
	begin_upload: proc(data: rawptr) -> rawptr,
	upload_mesh: proc(data: rawptr, upload: rawptr, mesh: ^CpuMesh) -> MeshHandle,
	end_upload: proc(data: rawptr, upload: rawptr),
	replace_mesh: proc(data: rawptr, handle: MeshHandle, mesh: ^CpuMesh),
}

renderer_api_begin_upload :: proc(api: RendererApi) -> rawptr {
	assert(api.begin_upload != nil)
	return api.begin_upload(api.data)
}

renderer_api_upload_mesh :: proc(api: RendererApi, upload: rawptr, mesh: ^CpuMesh) -> MeshHandle {
	assert(api.upload_mesh != nil)
	return api.upload_mesh(api.data, upload, mesh)
}

renderer_api_end_upload :: proc(api: RendererApi, upload: rawptr) {
	assert(api.end_upload != nil)
	api.end_upload(api.data, upload)
}

renderer_api_replace_mesh :: proc(api: RendererApi, handle: MeshHandle, mesh: ^CpuMesh) {
	assert(api.replace_mesh != nil)
	api.replace_mesh(api.data, handle, mesh)
}
