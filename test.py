import cv2
import matplotlib.pyplot as plt

def show_image_with_matplotlib(title, frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Пример использования
camera_index = 0
cap = cv2.VideoCapture(camera_index)

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Не удалось захватить изображение с камеры {camera_index}.")
        break
    
    show_image_with_matplotlib(f'Камера {camera_index}', frame)
    
    # Прекращение по нажатию 'q' в консоли (это для отображения с использованием cv2.imshow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
