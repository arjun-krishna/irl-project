from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Demonstration:
    observations = []
    actions = []
    depth_maps = []

    def __init__(self):
        pass

    def add(self, obs: np.ndarray, action, depth_map: np.ndarray = None):
        self.observations.append(obs.copy())
        self.actions.append(action)  
        if depth_map is not None:
            self.depth_maps.append(depth_map.copy())
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.depth_maps = []

    def getDict(self) -> Dict[str, List]:
        return {
            'observations': self.observations,
            'actions': self.actions,
            'depth_maps': self.depth_maps
        }
    
    def setDict(self, dict: Dict[str, List]):
        for field in ['observations', 'actions', 'depth_maps']:
            assert field in dict
        
        self.observations = dict['observations']
        self.actions = dict['actions']
        self.depth_maps = dict['depth_maps']

    def play(self):
        show_depth_maps = len(self.depth_maps) > 0
        if show_depth_maps:
            fig, (ax_obs, ax_dm) = plt.subplots(1, 2)
            ax_obs.set_title('Observation')
            ax_dm.set_title('Depth Map')
        else:
            fig, ax_obs = plt.subplots()
        ims = []
        for i in range(len(self.observations)):
            im = ax_obs.imshow(self.observations[i], animated=True)
            if i == 0:
                ax_obs.imshow(self.observations[i])
            # ims.append([im])
            title = plt.text(15.0, 5.0,str(self.actions[i]), ha="center",va="bottom",color=[0,0,0],
                     transform=ax_obs.transAxes, fontsize="large")
            text = ax_obs.text(15.0, 5.0, str(self.actions[i]))
            
            if show_depth_maps:
                dm = ax_dm.imshow(self.depth_maps[i], animated=True)
                if i == 0:
                    ax_dm.imshow(self.depth_maps[i])
                ims.append([im, title, text, dm])
            else:
                ims.append([im, title, text])
        anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        plt.show()