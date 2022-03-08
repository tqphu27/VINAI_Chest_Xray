# Kiến trúc mạng YOLO
1. Yolov1
 Ý tưởng chính của Yolov1 là chia ảnh thành một lưới các ô(grid cell) với kích thước SxS (mặc định 7x7).
 Với mỗi grid cell, mô hình sẽ đưa ra dự đoán cho B bounding box. Ứng với mỗi bõ trong B bounding box này sẽ là 5 tham số x,y,w,h,confidence lần lượt là tọa độ tâm(x,y), chiều rộng, chiều cao và dộ tự tin của dự đoán. Với mỗi grid cell trong lưới SxS kia, mô hình cũng dự đoán xác suất rơi vào mỗi class.
 
src="![image](https://user-images.githubusercontent.com/90370260/157260671-7f163747-aadc-4943-a06f-b99b305e3d83.png)
          
 YOLOv1 không có các hạn chế trong việc dự đoán vị trí của bounding box. Khi các trọng số được khởi tạo ngẫu nhiên, bounding box có thể được dự đoán ở bất kỳ đâu trong ảnh. Điều này khiến mô hình không ổn định trong giai đoạn đầu của quá trình huấn luyện. Vị trí của bounding box có thể ở rất xa so với vị trí của grid cell.
*Nhược điểm:
  Yolov1 áp đặt các rằng buộc vể không gian trên những bounding box, mỗi grid cell chỉ có thể predict rất ít bounding box (B) và duy nhất một class => hạn chế khả năng nhận biết một số object nằm gần nhau, cũng như đối với các object có kích thước nhỏ.
  Ngoài ra, trong quá trình training, loss function không có sự đánh giá riêng biệt giữa error của bounding box kích thuociws nhỏ so với error của bounding box kích thước lớn.*
  
2. Yolov2
 Kĩ thuật Batch Normalization được đưa vào sau tất cả các lớp convolution của YOLOv2.
 Yolov2 được huấn luyện với hai pha. Pha đầu sẽ huần luyện một mạng classifier với ảnh đầu vào kích thước nhỏ và pha sau sẽ loại vỏ lớp fully connected và sử dụng mạng classifier này như phần khung xương (backbone) để huấn luyện mạng detection.
 Ở pha sau YOLO trước hết finetune mạng backbone dưới ảnh đầu vào kích thước lớn hơn là 448x448, để mạng "quen" dần với kích thước ảnh đầu vào lớn, sau đó mới sử dụng kết quả này để huấn luyện cho quá trình detection. Điều này giúp tăng mAP của YOLOv2 lên khoảng 4%.
 Trong Yolov2, tác giả loại bỏ lớp fully connected ở giữa mạng và sử dụng kiến trúc anchorbox để predict các bounding box. Việc dự đoán các offset so với anchorbox sẽ dễ dàng hơn nhiều so với dự đoán tọa độ bounding box.
 Thay vì phải chọn anchorbox bằng tay, YOLOv2 sử dụng thuật toán k-means để đưa ra các lựa chọn anchorbox tốt nhất cho mạng. Việc này tạo ra mean IoU tốt hơn.
 YOLOv2 sử dụng hàm sigmoid ( ) để hạn chế giá trị trong khoảng 0 đến 1, từ đó có thể hạn chế các dự đoán bounding box ở xung quanh grid cell, từ đó giúp mô hình ổn định hơn trong quá trình huấn luyện.
 Faster R-CNN và SSD đưa ra dự đoán ở nhiều tầng khác nhau trong mạng để tận dụng các feature map ở các kích thước khác nhau. YOLOv2 cũng kết hợp các feature ở các tầng khác nhau lại để đưa ra dự đoán, cụ thể kiến trúc nguyên bản của YOLOv2 kết hợp feature map 26x26 lấy từ đoạn gần cuối với feature map 13x13 ở cuối để đưa ra các dự đoán. Cụ thể là các feature map này sẽ được ghép vào nhau (concatenate) để tạo thành một khối sử dụng cho dự đoán.
src="![image](https://user-images.githubusercontent.com/90370260/157259975-3a68c2a6-7274-4e90-900d-aea06e732cd0.png)
 Điểm cải tiến của YOLOv2 còn phải kể đến backbone mới có tên Darknet-19. Mạng này bao gồm 19 lớp convolution và 5 lớp maxpooling tạo ra tốc độ nhanh hơn phiên bản YOLO trước.

3. Yolov3
 

4. Yolov3
5. Yolov5
# VINAI_Chest_Xray

- IoU: phần giao lớn và phần hợp nhỏ.
- Precision – đại diện cho độ tin cậy của model: Nó sẽ cho biết rằng trong những cái model dự đoán là Positive thì có bao nhiêu % là Positive thật.
- Recall – đại diện cho độ nhạy của model: Nó sẽ cho biết model có thể tóm đúng được bao nhiêu Positive trong dữ liệu được cho.
- mAP: Trong bài này thì đơn giản mAP là Mean Average Precision là trung bình cộng giá trị AP của các class khác nhau.

Project được chia thành 2 phần chính:
- Xử lý dữ liệu (Gồm 14 class).
- Huấn luyện model bằng cách sử dụng mô hình yolo5.

 1. Xử lý dữ liệu 
- Dữ liệu cuộc thi khá lớn do ở dạng DICOM nên ở đây mình sử dụng một bộ dataset đã được convert sang PNG. [Here](https://www.miai.vn/thu-vien-mi-ai/.)
- Dữ liệu gồm 3 file:
  +   Một file csv- chứa thông tin về ảnh, nhãn của các vùng chuẩn đoán.
  +   Một folder train: chứa các ảnh để train model.
  +   Một foler test: Chứa các ảnh để test model.
- Tiền xử lý dữ liệu:(preprocess.py)
  + Đầu tiên ta sẽ sử dụng file csv để lấy tên file và tọa độ các box, nhãn của các bệnh chuẩn đoán.
  + Tiếp tục ta sẽ đọc file ảnh để biết kích thước ảnh.
  + Convert sang YOLO format và ghi vào folder train.
- Chia dữ liệu thành train và val theo cấu trúc YoloV5 (split.py)

2. Huấn luyện mô hình 
- Chọn mô hình để huấn luyện(Ở đây mình chọn 2 mô hình yolo5m và yolo5x).
- Trong mỗi mô hình, vào file yaml tương ứng và sửa thông số nc=14(trùng với số class)
- Tạo một file data.yml trong models: 
<p align="center"> 
    <img src="https://user-images.githubusercontent.com/90370260/155838525-b26c0749-67a8-44a9-bc4d-e570a404e0a5.png" alt="alternate text">
 </p>
 
- Sau đó training:
  + Theo model yolo5m: [Notebook](https://colab.research.google.com/drive/1axv9C87HVGcVCnFdxPSfeZ5TAw6JU63q?authuser=2#scrollTo=yeu77eYsOOCc)
  + Theo model yolo5x: [Notebook](https://colab.research.google.com/drive/1cXx2pt9JLaXPuN40WDBOZl2fxf5arU9T?authuser=4&hl=vi)

# Kết quả cuối cùng được lưu trong đường dẫn sau:
 - Mô hình Yolo5m: [Drive](https://drive.google.com/drive/folders/1cwxqKLQl_a9UINxOQn3RGI_ip0JSmYvl?usp=sharing)
 - Mô hình Yolo5x: [Drive](https://drive.google.com/drive/u/4/folders/12q2rYtIYcN4FJEEtUDnl6i6jrXzTbG_V)
- Nhận xét: mAP của cả hai mô hình có giá trị xấp xỉ nhau(khoảng 0.245), không được cao cho lắm :(
