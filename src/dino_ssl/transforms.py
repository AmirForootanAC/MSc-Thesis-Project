import random
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps

class GaussianBlur:
    """اعمال بلور گوسی با شعاع تصادفی (بسیار مهم برای DINO)"""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.prob:
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        return img

class Solarization:
    """معکوس کردن پیکسل‌های روشن (کمک به درک بهتر لبه‌های دندان و استخوان)"""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

class DINORadiographTransforms:
    """
    تولید کراپ‌های Multi-Crop برای DINO
    شامل 2 کراپ سراسری (Global) برای Teacher و Student
    شامل چند کراپ محلی (Local) فقط برای Student
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        global_crops_size=224,
        local_crops_size=96,
        local_crops_number=4, # تعداد تکه‌های کوچکی که از هر عکس بریده می‌شود
    ):
        # آگمنتیشن‌های رنگ و نور متناسب با X-Ray (حذف رنگ، تمرکز روی کنتراست)
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0
        )
        
        # ۱. تنظیمات کراپ سراسری اول (Global 1)
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=1.0), # X-ray همیشه خاکستری است
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # ۲. تنظیمات کراپ سراسری دوم (Global 2) - کمی متفاوت‌تر
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=1.0),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # ۳. تنظیمات کراپ‌های محلی (Local) - فقط روی جزئیات ریز فک زوم می‌کنند
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=1.0),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, image):
        crops = []
        # اضافه کردن 2 کراپ سراسری
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        
        # اضافه کردن کراپ‌های محلی
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
            
        return crops