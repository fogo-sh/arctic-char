#+test
package game

import "core:sync"

// Box3D's world operations are isolated by world handle in runtime code, but the
// vendored binding is not safe to exercise from multiple Odin test workers.
physics_test_mutex: sync.Mutex

physics_test_lock :: proc "contextless" () {
	sync.mutex_lock(&physics_test_mutex)
}

physics_test_unlock :: proc "contextless" () {
	sync.mutex_unlock(&physics_test_mutex)
}
