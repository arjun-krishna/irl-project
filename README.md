# IRL - Project

### Environments:
- MiniWorld-Hallway-v0

- MiniWorld-OneRoom-v0
- MiniWorld-OneRoomS6-v0
- MiniWorld-OneRoomS6Fast-v0 

- MiniWorld-TMaze-v0
- MiniWorld-TMazeLeft-v0
- MiniWorld-TMazeRight-v0

- MiniWorld-YMaze-v0
- MiniWorld-YMazeLeft-v0
- MiniWorld-YMazeRight-v0

- MiniWorld-Maze-v0
- MiniWorld-MazeS3-v0
- MiniWorld-MazeS3Fast-v0
- MiniWorld-MazeS2-v0

- MiniWorld-FourRooms-v0

- MiniWorld-Sidewalk-v0

### File Desc
- manual_control.py: control with keyboard actions
- record-demonstration.py: record and store expert demonstrations

### Snippet to replay recorded demos
```python
from core.utils import load_demo
d = load_demo('/*SPECIFY PATH*/')
d.play() # loads a matplotlib looped animation
```

### Training Behavior Cloning
```bash
$> python -m agents.behavior_clone
```

### Running a trained model (based on a fixed architecture defined in agents/net.py)
```bash
$> python -m agents.run_model --model_path {specify the model.pt to run}
```

*Note*: that top-view is not accounted for and the scripts need more parametrization in terms of arguments

### Team can add datasets, models and demos on google drive here

https://drive.google.com/drive/folders/1IF5-IGnkqwf6dR63lHx6BIPFEPWPiVH-?usp=sharing

Reason - could connect with colab hopefully
*TODO* remove editable link when making repo public