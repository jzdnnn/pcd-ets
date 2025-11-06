extends Control

const MAIN_MENU_SCENE = "res://scenes/MainMenu.tscn"

func _ready():
	print("About screen loaded")

func _on_back_button_pressed():
	"""Return to main menu."""
	print("Returning to main menu...")
	get_tree().change_scene_to_file(MAIN_MENU_SCENE)
