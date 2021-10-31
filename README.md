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