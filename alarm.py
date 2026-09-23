import os
import pygame


class AlarmManager:
    def __init__(self, alarm_path="assets/alarm.wav"):
        self.available = False
        self.playing = False

        try:
            pygame.mixer.init()

            if os.path.exists(alarm_path):
                pygame.mixer.music.load(alarm_path)
                self.available = True
            else:
                print(f"[WARNING] Alarm file not found: {alarm_path}")

        except pygame.error as error:
            print(f"[WARNING] Audio initialization failed: {error}")

    def play(self):
        if not self.available:
            return

        if not self.playing:
            try:
                pygame.mixer.music.play(-1)
                self.playing = True
            except pygame.error as error:
                print(f"[WARNING] Could not play alarm: {error}")

    def stop(self):
        if not self.available:
            return

        if self.playing:
            try:
                pygame.mixer.music.stop()
                self.playing = False
            except pygame.error:
                pass

    def cleanup(self):
        self.stop()

        try:
            pygame.mixer.quit()
        except pygame.error:
            pass