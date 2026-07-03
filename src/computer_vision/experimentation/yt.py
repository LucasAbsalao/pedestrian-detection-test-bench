import yt_dlp

url = 'https://youtu.be/0YF8vecQWYs'

'''
To run via terminal:
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" -o "%(title)s.%(ext)s" "https://youtu.be/Yk-EXRmGKG0"
'''

ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',

    'outtmpl': '/Youtube/%(title).%(ext)s',

    'quiet': False
}

print("Starting Download...")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Download finished with success!")