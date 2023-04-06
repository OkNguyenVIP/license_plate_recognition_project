"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from home.views import index
from license_plate_recognition.views import license_plate_recognition, video_feed, stream_data

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('license-plate-recognition/', license_plate_recognition, name='license-plate-recognition'),
    path('video_feed/', video_feed, name='video_feed'),
    path('stream_data/', stream_data, name='stream_data'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

admin.site.site_title = "Quản trị - OkNguyen"
admin.site.site_header = "Quản trị"
