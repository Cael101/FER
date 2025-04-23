# %%
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# %%
img = cv2.imread('happyboy.jpg')  # อ่านภาพ
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลง BGR → RGB (เพราะ OpenCV ใช้ BGR แต่ matplotlib ใช้ RGB)

# %%
plt.imshow(img_rgb)  # แสดงภาพ
plt.axis('off')     # ปิดแกนพิกัด
plt.show()          # โชว์ภาพในหน้าต่างแยก (ถ้าใช้ใน Interactive Window อาจไม่จำเป็น)

# %%
predictions = DeepFace.analyze(img, actions=['emotion'])
print("Dominant emotion:", predictions[0]['dominant_emotion'])
type(predictions)


# %%
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(faceCascade.empty())
faces = faceCascade.detectMultiScale(gray,1.1,4)

    #วาดรูปสี่เหลี่ยมรอบๆใบหน้า
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')     # ปิดแกนพิกัด
    plt.show() 
# %%
font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
cv2.putText(img,
            predictions[0]['dominant_emotion'],
            (0,200),
            font, 2,
            (0,0,255),
            3,
            cv2.LINE_4);
# %%
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')     # ปิดแกนพิกัด
plt.show() 
# %%
