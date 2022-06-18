# Derin Öğrenme Teknolojisi ile Beyin Tümörü Tespiti ve Segmentasyonu

> **Matematik Mühendisliği Bitirme Çalışması kapsamında "Derin Öğrenme Teknolojisi ile Beyin Tümörü Tespiti ve Segmentasyonu" konusu ele alınmış olup, tümörü kolaylıkla ve yüksek doğrulukta tespit edebilen bir bilgisayar destekli tümör tespit sistemi geliştirilmiştir.**
---

## Genel Bilgiler
Beyin tümörü en yaygın ve tehlikeli tümör hastalıklarının başını çekmektedir. Erken teşhisi hayati önem taşımakta olup, tümörün ilerleyen evrelerde teşhisi hastanın ölümü ile sonuçlanabilmektedir. Bu sebeple beyin tümörünün teşhisi oldukça kritik ve önemli bir süreçtir. Bu çalışmada, beyin MR görüntülerini kullanarak tümörü hızlı, kolay ve hatasız tespit edebilen bir bilgisayar destekli tümör tespit sistemi geliştirilmiştir. Beyin tümörünün MR görüntülerinde tespit edilebilmesi için derin öğrenme teknolojisi ve görüntü işleme teknikleri kullanılmıştır. Veri seti github platformundan alınmış olup, [Beyin Tümörü Tespiti için Beyin MR Görüntüleri](https://github.com/rastislavkopal/brain-tumor-segmentation/tree/main/brain_tumor_data) veri seti kullanılmıştır. Veri seti test, eğitim (train) ve geliştirme (val) olmak üzere 3 kümeye ayrılmıştır. Eğitim kümesi 202 veri, geliştirme kümesi 61 veri ve test kümesi 39 veriden oluşmaktadır. Veriler tümör içeren MR görüntülerine ait olup, tümörün yerinin segmentasyon yöntemi ile detaylı olarak belirlenmesi hedeflenmiştir.

<img width="628" alt="image" src="https://user-images.githubusercontent.com/68853621/174295928-e44859e9-8a74-4fc4-adc7-33865512eb1c.png">

Derin öğrenme kütüphanesi olarak TensorFlow ve Keras kütüphaneleri kullanılmıştır. Python programlama dili ile çalışılmış olup, GPU için Google’ın sunduğu Colab kullanılmıştır. Tespit için gerekli olan öğrenme süreci, derin öğrenme teknolojisinin yapıtaşı olan yapay sinir ağları (YSA) çeşitlerinden *Maske Bölge Tabanlı Evrişimsel Sinir Ağları (Mask R-CNN)* ile öğrenme gerçekleştirilmiş olup yüksek başarım oranı hedeflenmiştir. Model olarak saniyede alınan kare hızı (FPS), çıkarım zamanı, başarım oranı ve Mask R-CNN ile çalışacak olması göz önüne alınmış olup *ResNet-101* derin öğrenme modeli kullanılmıştır. Modelin 15 tur ve her turda 100 iterasyon ile eğitilmiştir. **Ağırlık dosyası logs klasörü içerisinde bulunmaktadır.**


## Materyal ve Yöntem

Derin öğrenme, görüntü sınıflandırması ve tespiti için oldukça yaygın, kullanışlı ve verimli bir yöntemdir. Yapay Sinir Ağları (YSA), derin öğrenme ile görüntü sınıflandırması teknolojisinin temel tekniğidir. Derin öğrenme ile görüntü üzerinde tespit ve sınıflandırma işlemleri için YSA algoritmalarından Evrişimli Sinir Ağları (ESA) kullanılmaktadır. Bu çalışma için ESA algoritmalarından olan R-CNN (Bölge Tabanlı Evrişimli Sinir Ağları), Fast R-CNN, Faster R-CNN ve Mask R-CNN algoritmaları detaylı olarak araştırılmış olup, bu algoritmalar ile yapılmış olan çalışmalar incelenmiştir. İstenilen ve hedeflenen tespit süreci doğrultusunda yalnızca sınıflandırma yapılmak istenmemiş olup segmentasyon algoritmaları ve mantığı araştırılmıştır. Yapılan çalışmalar sonucunda sisteme verilen tümörlü MR görüntüsünün işlenerek tümör bölgesinin segmentasyon yapılarak tespit edilmesi ve işaretlenmesi planlanmıştır. [Bu doğrultuda yapılan çalışmalar incelenmiş olup, Mask R-CNN kullanıldığı gözlemlenmiş ve projede kullanımı kararlaştırılmıştır.](https://paperswithcode.com/paper/mask-r-cnn/review/?hl=49216)

<img width="414" alt="image" src="https://user-images.githubusercontent.com/68853621/174296464-d08a6e3c-ae2c-457d-b807-977c24781ce4.png">

<img width="472" alt="image" src="https://user-images.githubusercontent.com/68853621/174296500-505966da-1c7d-4e05-814d-443f6f16dc20.png">

Tümör tespit sistemi gerçek zamanlı bir sistem olmayıp, saniyede alınan kare hızı (FPS) bu çalışma için önemli değildir. Tümörün doğru tespiti başlıca hedef olup, araştırmalar bu doğrultuda sürdürülmüştür. Mask R-CNN algoritmasında omurga olarak ResNet50 veya ResNet101 kullanılmaktadır. Bu çalışmada öncelik yüksek doğruluk ve hassasiyet olduğundan, [yapılan literatür araştırmaları sonucu](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf?ref=) ResNet101’in doğruluk oranının daha yüksek olduğu gözlemlenmiş olduğundan Mask R-CNN ile omurga olarak ResNet101 kullanılmıştır. ResNet101, 101 katmanlı bir derin öğrenme modelidir fakat kullandığı artık bloklar sayesinde bir sonraki katmana aktarılan bilginin kaybolması/zarar görmesi problemini çözerek yüksek doğruluk sağlamaktadır. 

<img width="400" alt="image" src="https://user-images.githubusercontent.com/68853621/174297753-22fb311b-58ea-4e2d-9421-0b5252fa33df.png">


## Tümör Tespit Süreci ve Algoritma

244 MB büyüklüğündeki 15. epoch sonunda güncellenerek yedeği alınan son ağırlık dosyası Mask R-CNN klasörü altında açılan logs klasörüne dahil edilmiştir. Tahmin ve teşhis kodu için gerekli olan ağırlık dosyası (.h5) klasör yolu bu düzenden çekilmiştir. Tespit işlemi için gerekli olan görüntüleme fonksiyonları yazılmış olup kod çalıştırılmıştır.

<img width="300" alt="image" src="https://user-images.githubusercontent.com/68853621/174298136-47b2cec6-2871-4599-b730-cbf76846c4df.png">


## Beyin Tümörü Karar Destek Sistemi
---

**Başarılı bir şekilde çalıştığı gözlemlenen algoritmanın bir karar destek sistemine taşınarak kullanıcıya kolaylık sağlaması hedeflenmiştir. Bu hedef doğrultusunda Python programlama dilinin sunduğu IPywidgets kütüphanesi ve HTML kullanılarak basit bir arayüz geliştirilmiştir. Geliştirilen sistemin temel amacı uzman doktorun hastadan istediği MR görüntüsünün yüklenebileceği ve ardından uzman doktora MR görüntüsünün hatasız tümör tespiti ve segmentasyonu gerçekleştirilmiş çıktıyı sunduğu bir ara yüz görevi görmesidir.**

<img width="488" alt="image" src="https://user-images.githubusercontent.com/68853621/174298511-86e87398-4003-47a2-9aaa-66c59b288441.png">

**‘upload’ butonu aracılığıyla kullanıcıdan (doktor, radyolog vb.) aldığı MR görüntüsüne kullanıcının ‘Tümör Tespiti’ butonuna basmasıyla birlikte tümör tespiti ve segmentasyonu uygulayarak çıktıyı yüksek doğruluk, hassasiyet ve hız ile kullanıcıya sunmaktadır.**

<img width="488" alt="image" src="https://user-images.githubusercontent.com/68853621/174298710-fd870272-790e-43ed-bde2-512316b8ff17.png">

EKRAN ÇIKTISI
---

<img width="350" alt="image" src="https://user-images.githubusercontent.com/68853621/174298804-7b691654-d206-4ecf-875a-0e5562b169cb.png">




## NOT
---
[Ağırlık dosyası](https://www.kaggle.com/code/ruslankl/brain-tumor-detection-v2-0-mask-r-cnn/data)
