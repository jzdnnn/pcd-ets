extends Node

# UDP Configuration
var udp_server := PacketPeerUDP.new()
var udp_sender := PacketPeerUDP.new()  # For sending commands to server
var server_host = "127.0.0.1"
var server_port = 5005  # Port for receiving video data
var command_port = 5006  # Port for sending commands to server
var is_receiving = false

# Mustache styles
var mustache_styles = []
var current_style = ""

# UI References
@onready var connection_value = get_node("../MarginContainer/VBoxContainer/StatusBar/ConnectionStatus/ConnectionValue")
@onready var face_value = get_node("../MarginContainer/VBoxContainer/StatusBar/FaceCount/FaceValue")
@onready var fps_value = get_node("../MarginContainer/VBoxContainer/StatusBar/FPS/FPSValue")
@onready var start_button = get_node("../MarginContainer/VBoxContainer/ControlPanel/StartButton")
@onready var stop_button = get_node("../MarginContainer/VBoxContainer/ControlPanel/StopButton")
@onready var port_input = get_node("../MarginContainer/VBoxContainer/ControlPanel/PortInput")
@onready var mustache_selector = get_node("../MarginContainer/VBoxContainer/ControlPanel/MustacheSelector")

# Parameter sliders and labels
@onready var scale_factor_slider = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/ScaleFactorBox/ScaleFactorSlider")
@onready var scale_factor_value = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/ScaleFactorBox/ScaleFactorValue")
@onready var min_neighbors_slider = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/MinNeighborsBox/MinNeighborsSlider")
@onready var min_neighbors_value = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/MinNeighborsBox/MinNeighborsValue")
@onready var mustache_scale_slider = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/MustacheScaleBox/MustacheScaleSlider")
@onready var mustache_scale_value = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/MustacheScaleBox/MustacheScaleValue")
@onready var mustache_offset_slider = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/MustacheYOffsetBox/MustacheYOffsetSlider")
@onready var mustache_offset_value = get_node("../MarginContainer/VBoxContainer/ParametersPanel/ParametersMargin/ParametersVBox/MustacheYOffsetBox/MustacheYOffsetValue")

# Stats
var frame_count = 0
var last_time = 0.0
var current_fps = 0.0

# Signals
signal data_received(data)

func _ready():
	# Don't bind immediately - wait for Start button
	print("UDP Receiver initialized. Press Start to begin.")
	update_ui_state(false)

func _process(delta):
	if not is_receiving:
		return
	
	# Check for incoming packets
	if udp_server.get_available_packet_count() > 0:
		var packet = udp_server.get_packet()
		var json_string = packet.get_string_from_utf8()
		
		# Parse JSON
		var json = JSON.new()
		var parse_result = json.parse(json_string)
		
		if parse_result == OK:
			var data = json.get_data()
			emit_signal("data_received", data)
			
			# Update stats
			update_stats(data)
		else:
			print("JSON Parse Error: ", json.get_error_message())
	
	# Calculate FPS
	calculate_fps(delta)

func start_receiving():
	"""Start UDP server and begin receiving data."""
	if is_receiving:
		return
	
	# Get port from input
	var port = int(port_input.text)
	if port > 0:
		server_port = port
	
	# Bind UDP socket
	var err = udp_server.bind(server_port, server_host)
	if err != OK:
		print("Error binding UDP: ", err)
		connection_value.text = "Error"
		connection_value.modulate = Color(1, 0.3, 0.3)
		return
	
	is_receiving = true
	print("UDP Receiver started at %s:%d" % [server_host, server_port])
	
	update_ui_state(true)
	connection_value.text = "Connected"
	connection_value.modulate = Color(0.3, 1, 0.3)

func stop_receiving():
	"""Stop UDP server."""
	if not is_receiving:
		return
	
	udp_server.close()
	is_receiving = false
	print("UDP Receiver stopped")
	
	update_ui_state(false)
	connection_value.text = "Disconnected"
	connection_value.modulate = Color(1, 0.3, 0.3)
	face_value.text = "0"
	fps_value.text = "0"

func update_ui_state(receiving: bool):
	"""Update button states."""
	start_button.disabled = receiving
	stop_button.disabled = not receiving
	port_input.editable = not receiving

func update_stats(data: Dictionary):
	"""Update face count and mustache styles from received data."""
	if data.has("faces"):
		var face_count = data["faces"].size()
		face_value.text = str(face_count)
	
	# Update mustache styles dropdown
	if data.has("mustache_styles") and data.has("current_style"):
		var styles = data["mustache_styles"]
		var current = data["current_style"]
		
		# Only update if styles changed
		if styles != mustache_styles:
			mustache_styles = styles
			update_mustache_selector()
		
		# Update current style selection
		if current != current_style:
			current_style = current
			select_current_style()

func update_mustache_selector():
	"""Populate mustache selector dropdown with available styles."""
	mustache_selector.clear()
	for i in range(mustache_styles.size()):
		var style = mustache_styles[i]
		mustache_selector.add_item(style, i)
	print("Updated mustache selector with ", mustache_styles.size(), " styles")

func select_current_style():
	"""Select the current active style in the dropdown."""
	for i in range(mustache_styles.size()):
		if mustache_styles[i] == current_style:
			mustache_selector.select(i)
			break

func send_command_to_server(command: Dictionary):
	"""Send command to Python server via UDP."""
	var json_string = JSON.stringify(command)
	var packet = json_string.to_utf8_buffer()
	
	# Send to command port (5006), not data port (5005)
	udp_sender.set_dest_address(server_host, command_port)
	var err = udp_sender.put_packet(packet)
	if err != OK:
		print("Error sending command: ", err)
	else:
		print("Sent command to ", server_host, ":", command_port, " -> ", json_string)

func calculate_fps(delta: float):
	"""Calculate and display FPS."""
	frame_count += 1
	last_time += delta
	
	if last_time >= 1.0:
		current_fps = frame_count / last_time
		fps_value.text = str(int(current_fps))
		frame_count = 0
		last_time = 0.0

# Button signal handlers
func _on_start_pressed():
	"""Handle Start button press."""
	start_receiving()

func _on_stop_pressed():
	"""Handle Stop button press."""
	stop_receiving()

func _on_mustache_selected(index: int):
	"""Handle mustache style selection."""
	if index >= 0 and index < mustache_styles.size():
		var selected_style = mustache_styles[index]
		print("Selected mustache style: ", selected_style)
		
		# Send command to server to change style
		var command = {
			"command": "change_style",
			"style": selected_style
		}
		send_command_to_server(command)

# Parameter slider signal handlers
func _on_scale_factor_changed(value: float):
	"""Handle scale factor slider change."""
	scale_factor_value.text = "%.2f" % value
	send_parameter_update()

func _on_min_neighbors_changed(value: float):
	"""Handle min neighbors slider change."""
	min_neighbors_value.text = str(int(value))
	send_parameter_update()

func _on_mustache_scale_changed(value: float):
	"""Handle mustache scale slider change."""
	mustache_scale_value.text = "%.2f" % value
	send_parameter_update()

func _on_mustache_y_offset_changed(value: float):
	"""Handle mustache offset slider change."""
	mustache_offset_value.text = "%.2f" % value
	send_parameter_update()

func send_parameter_update():
	"""Send all current parameter values to server."""
	var command = {
		"command": "update_parameters",
		"parameters": {
			"scale_factor": scale_factor_slider.value,
			"min_neighbors": int(min_neighbors_slider.value),
			"mustache_scale": mustache_scale_slider.value,
			"mustache_y_offset": mustache_offset_slider.value,
			"smoothing_factor": 0.3
		}
	}
	send_command_to_server(command)
	print("Sent parameter update: scale_factor=", scale_factor_slider.value, 
		" min_neighbors=", int(min_neighbors_slider.value),
		" mustache_scale=", mustache_scale_slider.value,
		" mustache_y_offset=", mustache_offset_slider.value)

func _exit_tree():
	"""Cleanup on exit."""
	if is_receiving:
		stop_receiving()
	udp_sender.close()
