"""
30.10.2022
Kumanda sınıfı 3 fonskiyon ekleyiniz.
85,93,101 numaralı kod satırlarlında bulunmaktadır.
"""



import time
import random



class Kumanda():

    def __init__(self, tv_durum="Kapalı",tv_ses=0,kanal_listesi=["TRT"],kanal="TRT"):

        self.tv_durum=tv_durum
        self.tv_ses=tv_ses
        self.kanal_listesi=kanal_listesi
        self.kanal= kanal

    def tv_ac(self):

        if (self.tv_durum  == "Kapalı"):
            print("Tv açılıyor")
            self.tv_durum="Açık"
        else:
            print("Tv zaten acik")

    def tv_kapat(self):
        
        if (self.tv_durum  == "Kapalı"):
            print("Tv zaten kapalı")
        
        else:
            print("Tv zaten kapaniyor")
            self.tv_durum="Kapalı"
    

    def ses_ayari(self):

        while True:

            cevap=input("Sesi artır :>\nSesi azalt: <\nÇıkış: q")

            if (cevap=='>'):
                if self.tv_ses>=100:
                    print("Max Ses")
                else:
                    self.tv_ses+=1
                print("Tv Sesi: {}".format(self.tv_ses))
            
            elif (cevap=="<"):
                if (self.tv_ses<=0):
                    print("Min ses")
                else:
                    self.tv_ses-=1
                print("tv_ses: {}",format(self.tv_ses))
            elif (cevap=="q"):
                break
            else:
                print("hatalı tuşlama")

    def kanal_ekle(self,kanal):
        if kanal in self.kanal_listesi:
            print("Kanal zaten var")
        else:
            
            print("Kanal ekleniyor...")
            time.sleep(1)
            

            self.kanal_listesi.append(kanal)

            print("Kanal listesi: {}".format(self.kanal_listesi))

    def kanal_sec(self):

        rastgele=random.randint(0,len(self.kanal_listesi)-1)

        self.kanal=self.kanal_listesi[rastgele]
        print(self.kanal)
    
    #################### 1 ####################
    def kanal_sil(self, kanal):
        if kanal not in self.kanal_listesi:
            print("Sectiginiz kanal bulunamadi")
        else:
            self.kanal_listesi.remove(kanal)
            print("Sectiginiz kanal silindi. Guncel kanal listesi : ", self.kanal_listesi)
            
    #################### 2 ####################        
    def kanal_ismi_degistir(self, kanal1, kanal2):
        if kanal1 not in self.kanal_listesi:
            print("Sectiginiz kanal bulunamadi")
        else:
            self.kanal_listesi[self.kanal_listesi.index(kanal1)] = kanal2
            print("Kanal ismi degistirildi. Guncel kanal listesi : ", self.kanal_listesi)  
            
    #################### 3 ####################
    #Aynı zamanda sorgulanması istenen kanal ismi tam verilmemişse "Bunu mu istediniz?" diye verilen stringle 
    # başlayan çıktıları verir.
    def kanal_sorgula(self, kanal):
        if kanal in self.kanal_listesi:
            print("Bu kanal mevcut.")
        if not any(filter(lambda x: x.startswith(kanal), self.kanal_listesi)):
            print("Bu isimde bir kanal mevcut değil.")
        else:
            self.kanallar = [i for i in self.kanal_listesi if i.startswith(kanal)]
            print("Bu kanalları mı demek istediniz ? : \n", self.kanallar)
            
    
    
    
    def __len__(self):
        return len(self.kanal_listesi)
    
    def __str__(self):
        return "Tv_durum: {}\ntv_ses: {}\nkanal listesi: {}\nkanal:{}".format(self.tv_durum,self.tv_ses,self.kanal_listesi,self.kanal)

print("""

        1. TV aç
        2. TV kapat
        3. Ses Ayarları
        4. Kanal Ekle
        5. Açık Kanalı Öğren
        6. Kanal Sayısı
        7. TV Bilgileri
        8. Kanal Sil
        9. Kanal İsmi Değiştir
        10. Kanal sorgula

Çıkmak için q'ya bas.

""")
kumanda=Kumanda()

while True:

    islem= input("İslem seçiniz ")

    if islem =="q":
        print("Sonlandırılıyor")
     
    elif islem =="1":
        kumanda.tv_ac()
    
    elif islem=="2":
        kumanda.tv_kapat()

    elif islem=="3":
        kumanda.ses_ayari()
    
    elif islem=="4":

        kanal_isimleri=input("Kanal isimlerini lütfen , ile ayırarak giriniz. ")

        x=kanal_isimleri.split(",")

        for i in x:
            kumanda.kanal_ekle(i)
    
    elif islem=="5":
        kumanda.kanal_sec()

    elif islem=="6":
        print("Kanal sayısı: ",len(kumanda))
    
    elif islem=="7":
        print(kumanda)
        
    elif islem=="8":
        kanal_ismi=input("Silmek istediginiz kanal ismini giriniz : ")
        kumanda.kanal_sil(kanal_ismi)
        
    elif islem=="9":
        kanal_ismi=input("Değiştirmek istediğini kanalı ve yeni adını ',' ile ayrırarak giriniz:  ")
        x = kanal_ismi.split(",")
        kumanda.kanal_ismi_degistir(x[0], x[1])
        
    elif islem=="10":
        kanal_ismi=input("Sorgulamak istediğiniz kanal adını giriniz : ")
        kumanda.kanal_sorgula(kanal_ismi)
    
    else:
        print("hatalı Tuşlama")



    
