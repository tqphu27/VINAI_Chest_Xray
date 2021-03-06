# Kiến trúc mạng YOLO
1. Yolov1
 Ý tưởng chính của Yolov1 là chia ảnh thành một lưới các ô(grid cell) với kích thước SxS (mặc định 7x7).
 Với mỗi grid cell, mô hình sẽ đưa ra dự đoán cho B bounding box. Ứng với mỗi bõ trong B bounding box này sẽ là 5 tham số x,y,w,h,confidence lần lượt là tọa độ tâm(x,y), chiều rộng, chiều cao và dộ tự tin của dự đoán. Với mỗi grid cell trong lưới SxS kia, mô hình cũng dự đoán xác suất rơi vào mỗi class.
 
![image](https://user-images.githubusercontent.com/90370260/157260671-7f163747-aadc-4943-a06f-b99b305e3d83.png)
          
 YOLOv1 không có các hạn chế trong việc dự đoán vị trí của bounding box. Khi các trọng số được khởi tạo ngẫu nhiên, bounding box có thể được dự đoán ở bất kỳ đâu trong ảnh. Điều này khiến mô hình không ổn định trong giai đoạn đầu của quá trình huấn luyện. Vị trí của bounding box có thể ở rất xa so với vị trí của grid cell.
 
*Nhược điểm:
  Yolov1 áp đặt các rằng buộc vể không gian trên những bounding box, mỗi grid cell chỉ có thể predict rất ít bounding box (B) và duy nhất một class => hạn chế khả năng nhận biết một số object nằm gần nhau, cũng như đối với các object có kích thước nhỏ.
  
  Ngoài ra, trong quá trình training, loss function không có sự đánh giá riêng biệt giữa error của bounding box kích thước nhỏ so với error của bounding box kích thước lớn.*
  
2. Yolov2

 Kĩ thuật Batch Normalization được đưa vào sau tất cả các lớp convolution của YOLOv2.
 Yolov2 được huấn luyện với hai pha. Pha đầu sẽ huần luyện một mạng classifier với ảnh đầu vào kích thước nhỏ và pha sau sẽ loại vỏ lớp fully connected và sử dụng mạng classifier này như phần khung xương (backbone) để huấn luyện mạng detection.
 
 Ở pha sau YOLO trước hết finetune mạng backbone dưới ảnh đầu vào kích thước lớn hơn là 448x448, để mạng "quen" dần với kích thước ảnh đầu vào lớn, sau đó mới sử dụng kết quả này để huấn luyện cho quá trình detection. Điều này giúp tăng mAP của YOLOv2 lên khoảng 4%.
 
 Trong Yolov2, tác giả loại bỏ lớp fully connected ở giữa mạng và sử dụng kiến trúc anchorbox để predict các bounding box. Việc dự đoán các offset so với anchorbox sẽ dễ dàng hơn nhiều so với dự đoán tọa độ bounding box.
 Thay vì phải chọn anchorbox bằng tay, YOLOv2 sử dụng thuật toán k-means để đưa ra các lựa chọn anchorbox tốt nhất cho mạng. Việc này tạo ra mean IoU tốt hơn.
 
 YOLOv2 sử dụng hàm sigmoid ( ) để hạn chế giá trị trong khoảng 0 đến 1, từ đó có thể hạn chế các dự đoán bounding box ở xung quanh grid cell, từ đó giúp mô hình ổn định hơn trong quá trình huấn luyện.
 
 Faster R-CNN và SSD đưa ra dự đoán ở nhiều tầng khác nhau trong mạng để tận dụng các feature map ở các kích thước khác nhau. YOLOv2 cũng kết hợp các feature ở các tầng khác nhau lại để đưa ra dự đoán, cụ thể kiến trúc nguyên bản của YOLOv2 kết hợp feature map 26x26 lấy từ đoạn gần cuối với feature map 13x13 ở cuối để đưa ra các dự đoán. Cụ thể là các feature map này sẽ được ghép vào nhau (concatenate) để tạo thành một khối sử dụng cho dự đoán.
 
![image](https://user-images.githubusercontent.com/90370260/157259975-3a68c2a6-7274-4e90-900d-aea06e732cd0.png)

 Điểm cải tiến của YOLOv2 còn phải kể đến backbone mới có tên Darknet-19. Mạng này bao gồm 19 lớp convolution và 5 lớp maxpooling tạo ra tốc độ nhanh hơn phiên bản YOLO trước.

3. Yolov3
 YOLOv3 có kiến trúc khá giống YOLOv2. Tác giả đã thêm các cải tiến mới trong các nghiên cứu gần đây vào YOLOv2 để tạo ra YOLOv3. Các cải tiến đó bao gồm:
  + Logistic regression cho confidence score: Yolov3 predict độ tự tin của bounding box sử dụng logistic regression.
  
  + Thay softmax bằng các logistic classifier rời rạc: Việc này cho hiệu quả tốt hơn nếu các label không "multually exclusive" , tức là có thể có đối tượng cùng thuộc 2 hay nhiều class khác nhau.
  
  + Backbone mới - Darknet-53: Backbone được thiết kế lại với việc thêm các residual blocks (kiến trúc sử dụng trong ResNet).
  
  + Backbone mới - Darknet-53: Backbone được thiết kế lại với việc thêm các residual blocks (kiến trúc sử dụng trong ResNet).
  
  + Skip-layer concatenation: YOLOv3 cũng thêm các liên kết giữa các lớp dự đoán. Mô hình upsample các lớp dự đoán ở các tầng sau và sau đó concatenate với các lớp dự đoán ở các tầng trước đó. Phương pháp này giúp tăng độ chính xác khi predict các object nhỏ.
  
   ![image](https://user-images.githubusercontent.com/90370260/157262030-0b07fb8a-102e-4571-939f-13e84eb4fdd7.png)

4. Yolov4

![image](https://user-images.githubusercontent.com/90370260/157271827-0a42da73-3a59-4551-95f0-5f5fa0c5c885.png)

Cấu trúc của v4 được tác giả chia làm bốn phần:

  + Backbone (xương sống)
  
  + Neck (cổ)
  
  + Dense prediction(dự đoán dày đặc) - sử dụng các one-stage-detection như YOLO hoặc SSD.
  
  + Sparese Prediction (dự đoán thưa thớt) - sử dụng các two- stage- detection như RCNN

 ![image](https://user-images.githubusercontent.com/90370260/157272362-16d51e44-0768-4ffb-8c62-91e705a9587d.png) 

- Backbone(Xương sống)- Trích xuất đặc trưng

 Mạng xương sống cho nhận dạng vật thể thường được đào tạo trước thông qua bài toán phân loại ImageNet.
 
 - CPSDarknet53
   CSP(Cross- Stage- Partial connections) có nguồn gốc từ kiến trúc DenseNet sử dụng đầu vào trước đó và nối nó với đầu vào hiện tại trước khi chuyển vào Dense layer.
   Nó có nhiệm vụ chia đầu vào của khối thành 2 phần, một phần sẽ qua các khối chập, và phần còn lại thì không.
   => Sau đó 2 phần sẽ được cộng lại và đưa vào khối tiếp theo.
   => Ý tưởng ở đây là loại vỏ các nút thắt tình toán trong DenseNet và cải thiện việc học bằng accsh chuyển phiên bản chưa chỉnh sửa của feature maps.
   
   ![image](https://user-images.githubusercontent.com/90370260/157361770-1d4c098a-ad20-468b-9fbd-ec0e82f7f2be.png)

  DenseNet (Dense connected convolutional network) là một trong những network mới nhất cho visual object recognition. Nó cũng gần giống Resnet nhưng có một vài điểm khác biệt .
  => Densenet có cấu trúc gồm các dense block và các transition layers. 
  => Được stack dense block- transition layers-dense block- transition layers như hình vẽ.
  => Với CNN truyền thống nếu chúng ta có L layer thì sẽ có L connection, còn trong densenet sẽ có L(L+1)/2 connection.( tức là các lớp phía trước sẽ được liên kết với tất cả các lớp phía sau nó).
  
  ![image](https://user-images.githubusercontent.com/90370260/157362202-9348a768-f922-47a4-8773-1410ec73b10f.png)

  => Darknet53: Yolov4 sử dụng CSPDarknet53 để làm backbone vì theo tác giả, CSPDarknet53 có độ chính xác trong task object detection cao hơn so với ResNet; và mặc dù ResNet có độ chính xác trong task classification cao hơn, hạn chế này có thể được cải thiện nhờ hàm activation Mish và một vài kỹ thuật sẽ được đề cập phía dưới.
 
 - Neck (phần cổ) - Tổng hợp đặc trưng.
 
  Neck có nhiệm vụ trộn và kết hợp các bản đồ đặc trưng(features map) đã học được thông qua quá trình trích xuất đặc trưng (backbone) và quá trình nhận dạng(Yolov3 gợi là Dense prediction).
  
  Với mỗi lần thực hiện detect với các kích thước ảnh rescale khác nhau tác giả đã thêm các luồng đi từ dưới lên và các luồng đi từ trên xuống vào cùng nhau theo từng hoặc được nối với nhau trước khi đưua vào head.
  => chứa được thông tin phong phú hơn.
  
  ![image](https://user-images.githubusercontent.com/90370260/157362701-38a2cf2b-65ba-44be-b2d0-390639b6379f.png)

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
