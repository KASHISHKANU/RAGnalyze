from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.documents import Document
import yt_dlp


def load_documents(url: str):
    if "youtube.com" in url or "youtu.be" in url:
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language=["en"]
            )
            return loader.load()

        except Exception:
            ydl_opts = {
                "quiet": True,
                "skip_download": True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            text = f"""
Title: {info.get('title', '')}

Description:
{info.get('description', '')}
"""

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": "youtube_description",
                        "url": url
                    }
                )
            ]

    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    return loader.load()
