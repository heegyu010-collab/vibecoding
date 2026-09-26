from .base import NY, Provider


def make_provider(settings) -> Provider:
    if settings.source == "yahoo":
        from .yahoo import YahooProvider
        return YahooProvider()
    if settings.source == "alpaca":
        from .alpaca import AlpacaProvider
        return AlpacaProvider(feed=settings.alpaca_feed)
    if settings.source == "synthetic":
        from .synthetic import SyntheticProvider
        return SyntheticProvider()
    raise ValueError(f"알 수 없는 데이터 소스: {settings.source}")


__all__ = ["NY", "Provider", "make_provider"]
