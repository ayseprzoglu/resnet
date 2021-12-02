# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:52:02 2021

@author: Casper-X400
"""

#VERİ ÇEKME LİNKLERİ 
import requests
from bs4 import BeautifulSoup
links = []
url = {
'https://www.zingat.com/satilik-konut?page=1',
'https://www.zingat.com/satilik-konut?page=2',
'https://www.zingat.com/satilik-konut?page=3',
'https://www.zingat.com/satilik-konut?page=4'
}

for i in url:
    r = requests.get(i)
    soup = BeautifulSoup(r.content, 'html5lib')
    soup.append(BeautifulSoup(r.content, 'html5lib'))
    for link in soup.find_all('a', class_='zl-card-inner'):
        links.append(link.get('href'))
links_final = list(dict.fromkeys(links))

links_final

"""
# EXCEL ÜZERİNDE DÜZENLEMELER YAPILARAK,LİNKLER OLUŞTURULDU

# VERİLERDEN GÖRÜNTÜ ÇEKME
"""

import requests
from bs4 import BeautifulSoup
images = []
url = {
'http://www.zingat.com/mekan-satiyor-yibo-okulu-civari-arakart-3-1-genis-daire-3896911i',
'http://www.zingat.com/son-daire-tek-tapu-yatirim-icin-en-ideal-daire-3479342i',
'http://www.zingat.com/alanya-mahmutlar-mah-fullesyali-fullaktiviteli-satilik-1-1-daire-3839373i',
'http://www.zingat.com/ultraluks-kira-garantili-esyali-rezidans-daire-1-625727i',
'http://www.zingat.com/erdek-ataturk-mah-sifir-esyali-deniz-manzarali-satilik-daire-3836502i',
'http://www.zingat.com/ortaklar-caddesi-uzerinde-mukemmel-konumlu-otoparkli-3-1-daire-3247040i',
'http://www.zingat.com/corlu-da-kelepir-masrafsiz-acill-satilik-dubleks-3830367i'
}


for i in url:
    r = requests.get(i)      #web sayfasını çek
    soup = BeautifulSoup(r.content, 'html5lib')
    soup.append(BeautifulSoup(r.content, 'html5lib'))  
    for link in soup.find_all('a', class_='gallery-item zoom-in-image'):
        images.append(link.get('href'))
images_final = list(dict.fromkeys(images))

b = 1;

for i in images_final:
    img = requests.get(i).content
    with open(r'C:\Users\Casper-X400\Desktop\veriler\ten{}.jpg'.format(b), 'wb') as handler:
        handler.write(img)
    b+=1;
        
#VERİLERİ YENİDEN BOYUTLANDIRMA

from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from PIL import Image

from PIL import Image
import glob

#Mutfak yeniden boyutlandır
inpath = r"C:\Users\Casper-X400\Desktop\mutfak"
outpath = r'C:\Users\Casper-X400\Desktop\resized.cekilen.görüntüler\mutfak'
images = []
resized_images =[]

for filename in glob.glob(r"C:\Users\Casper-X400\Desktop\mutfak\*.jpg"):
    print(filename)
    img = Image.open(filename)
    images.append(img)

for image in images:
    image=image.resize((512,512))
    resized_images.append(image)

for i,new in enumerate(resized_images):
    new.save('{}{}{}'.format('C:/Users/Casper-X400/Desktop/resized.cekilen.görüntüler\mutfak/img',i+1,'.jpg'))

#lavabo yeniden boyutlandır
inpath = r"C:\Users\Casper-X400\Desktop\lavabo"
outpath = r'C:\Users\Casper-X400\Desktop\resized.cekilen.görüntüler\lavabo'
images = []
resized_images =[]

for filename in glob.glob(r"C:\Users\Casper-X400\Desktop\lavabo\*.jpg"):
    print(filename)
    img = Image.open(filename)
    images.append(img)

for image in images:
    image=image.resize((512,512))
    resized_images.append(image)

for i,new in enumerate(resized_images):
    new.save('{}{}{}'.format('C:/Users/Casper-X400/Desktop/resized.cekilen.görüntüler\lavabo/img',i+1,'.jpg'))

#yatakodası yeniden boyutlandır
inpath = r"C:\Users\Casper-X400\Desktop\yatakodası"
outpath = r'C:\Users\Casper-X400\Desktop\resized.cekilen.görüntüler\yatakodası'
images = []
resized_images =[]

for filename in glob.glob(r"C:\Users\Casper-X400\Desktop\yatakodası\*.jpg"):
    print(filename)
    img = Image.open(filename)
    images.append(img)

for image in images:
    image=image.resize((512,512))
    resized_images.append(image)

for i,new in enumerate(resized_images):
    new.save('{}{}{}'.format('C:/Users/Casper-X400/Desktop/resized.cekilen.görüntüler\yatakodası/imgYatkOds',i+1,'.jpg'))

#salon yeniden boyutlandırma
inpath = r"C:\Users\Casper-X400\Desktop\salon"
outpath = r'C:\Users\Casper-X400\Desktop\resized.cekilen.görüntüler\salon'
images = []
resized_images =[]

for filename in glob.glob(r"C:\Users\Casper-X400\Desktop\salon\*.jpg"):
    print(filename)
    img = Image.open(filename)
    images.append(img)

for image in images:
    image=image.resize((512,512))
    resized_images.append(image)

for i,new in enumerate(resized_images):
    new.save('{}{}{}'.format('C:/Users/Casper-X400/Desktop/resized.cekilen.görüntüler\salon\imgSaln',i+1,'.jpg'))

# VERİ ARTIRMA

#MODEL







