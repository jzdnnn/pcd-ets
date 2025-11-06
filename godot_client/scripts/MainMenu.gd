extends Control

# Scene paths
const MAIN_SCENE = "res://scenes/Main.tscn"
const HELP_SCENE = "res://scenes/Help.tscn"
const ABOUT_SCENE = "res://scenes/About.tscn"

# Node references
@onready var background_image = $BackgroundContainer/BackgroundImage
@onready var start_button = $MainContainer/RightSide/MenuPanel/ButtonsContainer/StartButton
@onready var help_button = $MainContainer/RightSide/MenuPanel/ButtonsContainer/HelpButton
@onready var about_button = $MainContainer/RightSide/MenuPanel/ButtonsContainer/AboutButton
@onready var exit_button = $MainContainer/RightSide/MenuPanel/ButtonsContainer/ExitButton

func _ready():
	print("Main Menu loaded")
	# Focus on start button by default
	start_button.grab_focus()
	
	# Optional: Load background image via script
	# Uncomment line berikut dan sesuaikan path jika ingin load via code:
	# set_background_image("res://assets/backgrounds/menu_bg.png")

func _on_start_button_pressed():
	"""Start the application - transition to main scene."""
	print("Starting application...")
	get_tree().change_scene_to_file(MAIN_SCENE)

func _on_help_button_pressed():
	"""Open help/instructions screen."""
	print("Opening help screen...")
	get_tree().change_scene_to_file(HELP_SCENE)

func _on_about_button_pressed():
	"""Open about us screen."""
	print("Opening about screen...")
	get_tree().change_scene_to_file(ABOUT_SCENE)

func _on_exit_button_pressed():
	"""Exit the application."""
	print("Exiting application...")
	get_tree().quit()

func set_background_image(texture_path: String):
	"""
	Set custom background image.
	Background image should include the logo on the left side.
	Example: set_background_image("res://assets/backgrounds/menu_bg.png")
	"""
	var texture = load(texture_path)
	if texture:
		background_image.texture = texture
		print("Background image loaded: ", texture_path)
	else:
		print("Failed to load background image: ", texture_path)
