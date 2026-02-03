import numpy as np
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

class LineDataUnits(Line2D):
    #from https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1) 
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)
    
    
class CircleDataUnits(Circle):
    #from https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1) 
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def draw_causal_edge(ax,x, y, thickness,color):
    arrow = LineDataUnits(x, y, linewidth = thickness, color = color)
    
    ax.add_line(arrow)


def draw_population(ax, waveform,layer):
    waveform_color = ['orange','blue']
    layer = layer_space[layer-1]

    population = plt.Circle(xy = (waveform_space[waveform - 1],layer),
                            radius=node_radius, 
                            fc=waveform_color[waveform-1],
                            ec = 'Black', 
                            alpha = 1)
    ax.add_patch(population)
    
def draw_self_loop(ax,center, edge_radius, color, edge_thickness):
    edge = CircleDataUnits(xy = center,
                        radius=edge_radius, 
                        fill=False,
                        ec = color,
                        zorder = 0,
                        linewidth = edge_thickness)
    
    arrow_tip = LineDataUnits([center[0],center[0] + edge_radius/10],
                              [center[1], center[1]],
                              linewidth = edge_radius/10.3,
                              color = color)
    ax.add_patch(edge)
    ax.add_line(arrow_tip)
