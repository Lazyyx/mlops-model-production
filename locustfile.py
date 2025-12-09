# locustfile.py

from locust import HttpUser, task, between
import os

API_SECRET_TOKEN = os.getenv("APP_TOKEN", "fallback-token") 
IMAGE_FILE_PATH = "test_image.jpg"

class FaceDetectionUser(HttpUser):
    # simule un utilisateur qui attend entre 1 et 3 secondes entre chaque request
    wait_time = between(1, 3) 
    
    def on_start(self):
        self.headers = {
            "X-API-Key": API_SECRET_TOKEN
        }
        
    @task
    def detect_faces_task(self):
        try:
            with open(IMAGE_FILE_PATH, 'rb') as image_file:
                
                self.client.post(
                    "/api/detect", 
                    files={'file': image_file},
                    headers=self.headers,
                    name="/api/detect [POST]"
                )
        except FileNotFoundError:
            print(f"Erreur: Le fichier {IMAGE_FILE_PATH} est introuvable sur la machine cliente.")