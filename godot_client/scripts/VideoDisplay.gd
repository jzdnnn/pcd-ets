extends TextureRect

# Image processing
var image := Image.new()
var texture_img := ImageTexture.new()

# Face data
var faces = []
var fps = 0.0
var face_count = 0
var current_mustache_style = ""

func _ready():
	# Connect to UDP receiver
	var udp_receiver = get_node("/root/Main/UDPReceiver")
	if udp_receiver:
		udp_receiver.connect("data_received", Callable(self, "_on_data_received"))
		print("VideoDisplay: Connected to UDPReceiver")
	else:
		print("VideoDisplay: ERROR - UDPReceiver not found!")

func _on_data_received(data):
	# Update stats
	fps = data.get("fps", 0.0)
	faces = data.get("faces", [])
	face_count = data.get("face_count", 0)
	current_mustache_style = data.get("current_style", "")
	
	# DEBUG: Log received face count
	print("DEBUG VideoDisplay: Received ", faces.size(), " faces from server")
	
	# Decode and display frame
	var frame_base64 = data.get("frame", "")
	if frame_base64 != "":
		_display_frame(frame_base64)
	
	# Trigger redraw for bounding boxes
	queue_redraw()

func _display_frame(base64_string):
	# Decode base64 to bytes
	var image_data = Marshalls.base64_to_raw(base64_string)
	
	# Load JPEG from buffer
	var error = image.load_jpg_from_buffer(image_data)
	
	if error == OK:
		# Create texture from image
		texture_img = ImageTexture.create_from_image(image)
		self.texture = texture_img
	else:
		print("Error loading image: ", error)

func _draw():
	# DEBUG: Log draw call
	print("DEBUG VideoDisplay: _draw() called with ", faces.size(), " faces")
	
	# Calculate scale ratio between texture size and displayed size
	var scale_x = 1.0
	var scale_y = 1.0
	
	if texture != null and image != null:
		var texture_size = image.get_size()
		var display_size = self.size
		
		# Calculate actual scale based on stretch_mode
		if texture_size.x > 0 and texture_size.y > 0:
			scale_x = display_size.x / texture_size.x
			scale_y = display_size.y / texture_size.y
			
			# DEBUG: Log scaling info
			print("DEBUG VideoDisplay: texture_size=", texture_size, " display_size=", display_size)
			print("DEBUG VideoDisplay: scale_x=", scale_x, " scale_y=", scale_y)
	
	# Draw face bounding boxes with proper scaling
	for face in faces:
		# Scale coordinates to match display size
		var scaled_x = face.x * scale_x
		var scaled_y = face.y * scale_y
		var scaled_w = face.width * scale_x
		var scaled_h = face.height * scale_y
		
		var rect = Rect2(scaled_x, scaled_y, scaled_w, scaled_h)
		draw_rect(rect, Color.GREEN, false, 2.0)
		
		# Draw confidence text
		var confidence = face.get("confidence", 0.0)
		var text = "Face: %.1f%%" % (confidence * 100)
		draw_string(ThemeDB.fallback_font, Vector2(scaled_x, scaled_y - 5), text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, Color.GREEN)
	
	# Draw stats (top-left corner)
	var stats_y = 30
	draw_string(ThemeDB.fallback_font, Vector2(10, stats_y), "FPS: %.1f" % fps, HORIZONTAL_ALIGNMENT_LEFT, -1, 20, Color.YELLOW)
	draw_string(ThemeDB.fallback_font, Vector2(10, stats_y + 25), "Faces: %d" % face_count, HORIZONTAL_ALIGNMENT_LEFT, -1, 20, Color.YELLOW)
	
	# Draw current mustache style
	if current_mustache_style != "":
		draw_string(ThemeDB.fallback_font, Vector2(10, stats_y + 50), "Style: %s" % current_mustache_style, HORIZONTAL_ALIGNMENT_LEFT, -1, 20, Color.CYAN)
