package main

import game "../../src/game"
import json "core:encoding/json"
import "core:fmt"

main :: proc() {
	data, err := json.marshal(game.ENTITY_DEFINITIONS[:], {pretty = true, use_spaces = true, use_enum_names = true})
	if err != nil {
		fmt.eprintln("failed to marshal entity metadata:", err)
		return
	}
	defer delete(data)
	fmt.print(string(data))
}
