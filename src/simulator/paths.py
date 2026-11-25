from pathlib import Path

# Chemins des assets du simulateur, résolus depuis l'emplacement de ce fichier
# (indépendants du répertoire courant d'exécution)
project_root = Path(__file__).resolve().parents[2]
assets_folder_path = project_root / "assets" / "simulator"

robot_specs_file_path = assets_folder_path / "robots/2026/specs.json"
scene_parameters_file_path = assets_folder_path / "scenes/2027/params.json"
textures_folder_path = assets_folder_path / "textures"
robot_urdf_path = assets_folder_path / "robots/2026/robot.urdf"
