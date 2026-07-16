package engine

import "core:time"

performance_counter_now :: proc() -> u64 {
	return u64(time.tick_now()._nsec)
}

performance_elapsed_ms :: proc(start_counter: u64) -> f32 {
	end_counter := performance_counter_now()
	return f32(end_counter - start_counter) / 1_000_000.0
}
