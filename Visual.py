import numpy as np
import time, sys, math
import pygame
from out_Button import Button
from matplotlib import cm


class Visualizer:

    def __init__(self, display):
        self.mode_3D = True
        self.display = display

        self.HEIGHT  = self.display.GUI_height
        GUI_ratio = self.display.GUI_ratio

        self.HEIGHT = round(self.HEIGHT)
        self.WIDTH  = round(GUI_ratio*self.HEIGHT)
        self.y_ext = [round(0.05*self.HEIGHT), self.HEIGHT]
        self.cm = cm.tab20c
        #self.cm = cm.inferno

        self.choose_mode()

        self.add_outlines = 1
        self.outline_thickness = max(0.00002*self.HEIGHT, 1.25 / self.display.n_frequency_groups)

        color_indices = np.linspace(0, 1, self.display.n_frequency_groups)  # Create an index from 0 to 1

        # Use the tab20c colormap to assign colors based on the indices
        self.bar_colors = [list((255 * np.array(self.cm(i))[:3]).astype(int)) for i in color_indices]

        self.outline_colors = [(190,190,190)]*self.display.n_frequency_groups
        self.bar_colors = self.bar_colors[::-1]

        self.outline_features = [0]*self.display.n_frequency_groups
        self.group_max_energies  = np.zeros(self.display.n_frequency_groups)
        self.group_energies = self.display.group_energies
        self.bin_text_tags, self.bin_rectangles = [], []

        self._is_running = False

    def choose_mode(self):

        if self.mode_3D:
            self.bg_color           = (20, 20, 50)  
            self.decay_speed        = 0.10  #Vertical decay of slow bars
            self.bar_distance = 0
            self.avg_energy_height  = 0.1125
            self.move_fraction      = 0.0099

        else:
            self.bg_color           = (20, 20, 50)
            self.decay_speed        = 0.06
            self.bar_distance = int(0.2*self.WIDTH / self.display.n_frequency_groups)
            self.avg_energy_height  = 0.225

        self.bar_width = (self.WIDTH / self.display.n_frequency_groups) - self.bar_distance

        #Configure the bars:
        self.outlines, self.bars, self.bar_x_positions = [],[],[]
        for i in range(self.display.n_frequency_groups):
            x = int(i* self.WIDTH / self.display.n_frequency_groups)
            bar = [int(x), int(self.y_ext[0]), math.ceil(self.bar_width), None]
            outline = [int(x), None, math.ceil(self.bar_width), None]
            self.bar_x_positions.append(x)
            self.bars.append(bar)
            self.outlines.append(outline)

    def start(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.screen.fill(self.bg_color)

        if self.mode_3D:
            self.screen.set_alpha(255)
            self.prev_screen = self.screen
            
        pygame.display.set_caption('Audio 3D Visualization')
        self.bin_font = pygame.font.SysFont('comicsans', round(0.03*self.HEIGHT))

        for i in range(self.display.n_frequency_groups):      #the white freq axis
            if i == 0 or i == (self.display.n_frequency_groups - 1):
                continue
            if i % 8 == 0:
                f_centre = self.display.group_centres_f[i]
                text = self.bin_font.render('%d Hz' %f_centre, True, (255, 255, 255) , (self.bg_color))
                textRect = text.get_rect() 
                x = (i + 0.5) * (self.WIDTH / self.display.n_frequency_groups)  # Center the text in the bin
                y = 0.98 * self.HEIGHT 
                textRect.center = (int(x), int(y))
                self.bin_text_tags.append(text)
                self.bin_rectangles.append(textRect)

        self._is_running = True

        #Interactive components:
        self.button_height = round(0.05*self.HEIGHT)
        self.mode_button  = Button(text="2D/3D Mode Switch", right=self.WIDTH, top=0, width=round(0.12*self.WIDTH), height=self.button_height)



    def update(self):
        for event in pygame.event.get():
            if self.mode_button.click():
                self.mode_3D = not self.mode_3D
                self.choose_mode()

        self.mean_values = np.mean(self.display.group_energies)*8
        self.clip_mean_values = np.clip((self.mean_values-1000)/90000,0,4)+1
        self.group_energies = self.avg_energy_height*self.clip_mean_values * self.display.group_energies /self.mean_values
        
        
        if self.mode_3D:
            new_w, new_h = int((2+0.994)/3*self.WIDTH), int(0.994*self.HEIGHT)

            prev_screen = pygame.transform.scale(self.prev_screen, (new_w, new_h))

        self.screen.fill(self.bg_color)

        if self.mode_3D:
            new_pos = int(self.move_fraction*self.WIDTH - (0.0133*self.WIDTH)), int(self.move_fraction*self.HEIGHT)
            self.screen.blit(prev_screen, new_pos)

        self.plot_bars()


        if len(self.bin_text_tags) > 0:
            cnt = 0
            for i in range(self.display.n_frequency_groups):
                if i == 0 or i == (self.display.n_frequency_groups - 1):
                    continue
                if i % 8 == 0:
                    self.screen.blit(self.bin_text_tags[cnt], self.bin_rectangles[cnt])
                    cnt += 1

        self.mode_button.draw(self.screen)

        pygame.display.flip()


    def plot_bars(self):
        bars, outlines, new_outline_features = [], [], []
        local_height = self.y_ext[1] - self.y_ext[0]
        feature_values = self.group_energies[::-1]      #reverse

        for i in range(len(self.group_energies)):
            feature_value = feature_values[i] * local_height

            self.bars[i][3] = int(feature_value)

            # to avoid gray shadow in the animation
            if self.mode_3D:
                self.bars[i][3] = int(feature_value + 0.02*self.HEIGHT)

            self.decay = min(0.99, 1 - max(0,self.decay_speed * 60 / 30))
            outline_feature_value = max(self.outline_features[i]*self.decay, feature_value)
            new_outline_features.append(outline_feature_value)
            self.outlines[i][1] = int(self.bars[i][1] + outline_feature_value)
            self.outlines[i][3] = int(self.outline_thickness * local_height)

        for i, bar in enumerate(self.bars):
            pygame.draw.rect(self.screen,self.bar_colors[i],bar,0)

        if self.mode_3D:
                self.prev_screen = self.screen.copy().convert_alpha()
                self.prev_screen.set_alpha(self.prev_screen.get_alpha()*0.995)

        for i, outline in enumerate(self.outlines):
            pygame.draw.rect(self.screen,self.outline_colors[i],outline,0)

        self.outline_features = new_outline_features

        #Display everything:
        self.screen.blit(pygame.transform.rotate(self.screen, 180), (0, 0))


