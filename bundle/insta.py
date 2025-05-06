import instaloader
import requests
import re
import tempfile
import os

class InstaDownloader:
    def __init__(self):
        self.loader = instaloader.Instaloader()

    def _extract_shortcode(self, link):
        pattern = r"instagram\.com/(?:reel|p|tv)/([A-Za-z0-9_\-]+)"
        match = re.search(pattern, link)
        if not match:
            raise ValueError("Invalid Instagram URL")
        return match.group(1)

    def _download_media(self, url):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            suffix = os.path.splitext(url.split("?")[0])[-1]  # remove URL params
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            with open(temp_file.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return temp_file.name
        else:
            raise Exception("Failed to download media.")

    def download_post(self, insta_link):
        try:
            shortcode = self._extract_shortcode(insta_link)
            post = instaloader.Post.from_shortcode(self.loader.context, shortcode)

            media_url = post.video_url if post.is_video else post.url
            file_path = self._download_media(media_url)

            return file_path  # Path to the downloaded temp file

        except Exception as e:
            return f"Error: {str(e)}"


# Example usage
if __name__ == "__main__":
    downloader = InstaDownloader()
    file_path = downloader.download_post("https://www.instagram.com/p/DIbHbgQyBu2/?utm_source=ig_web_copy_link")
    print(f"Downloaded to: {file_path}")
    os.remove(file_path)
