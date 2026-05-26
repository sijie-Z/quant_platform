"""Cross-asset instrument abstraction.

Provides a unified instrument model for stocks, ETFs, futures, and options.
Eliminates hardcoded A-share assumptions (lot_size=100, multiplier=1, etc.)
by moving asset-specific parameters into per-instrument configuration.

Usage:
    universe = AssetUniverse.from_config(config)
    inst = universe.get("IF2406")          # futures contract
    notional = inst.notional(price, qty)   # price * multiplier * qty
    cost = inst.commission(price, qty)     # per-instrument cost model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class InstrumentType(StrEnum):
    """Supported asset classes."""
    STOCK = "stock"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    INDEX = "index"


@dataclass
class Instrument:
    """Unified instrument definition with asset-specific parameters.

    Attributes:
        symbol: Unique identifier (e.g. '600519', '510300', 'IF2406').
        asset_type: Asset class (stock/etf/future/option/index).
        exchange: Exchange code (SSE/SZSE/CFFEX/SHFE/etc.).
        multiplier: Contract multiplier (1 for stocks/ETFs, 300 for IF futures).
        tick_size: Minimum price increment (0.01 for stocks, 0.2 for IF futures).
        lot_size: Minimum trading unit (100 for A-shares, 1 for futures).
        margin_rate: Initial margin rate (1.0 for cash, 0.12 for futures).
        commission_rate: Per-trade commission rate (overrides global if set).
        commission_per_lot: Per-lot fixed commission (futures use this).
        stamp_tax_rate: Stamp tax rate (0.001 for stocks, 0 for futures/ETFs).
        t_plus: Settlement days (1 for A-shares, 0 for futures).
        price_limit: Daily price limit pct (0.10 for stocks, 0.20 for 创业板).
        underlying: Underlying symbol (for options/futures).
        expiry: Expiry date string (for options/futures).
        extra: Arbitrary extra metadata.
    """
    symbol: str = ""
    asset_type: InstrumentType = InstrumentType.STOCK
    exchange: str = "SSE"
    multiplier: float = 1.0
    tick_size: float = 0.01
    lot_size: int = 100
    margin_rate: float = 1.0
    commission_rate: float | None = None
    commission_per_lot: float = 0.0
    stamp_tax_rate: float | None = None
    t_plus: int = 1
    price_limit: float = 0.10
    underlying: str = ""
    expiry: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_derivative(self) -> bool:
        return self.asset_type in (InstrumentType.FUTURE, InstrumentType.OPTION)

    @property
    def is_equity(self) -> bool:
        return self.asset_type in (InstrumentType.STOCK, InstrumentType.ETF, InstrumentType.INDEX)

    def notional(self, price: float, quantity: int) -> float:
        """Calculate notional value: price * multiplier * quantity."""
        return price * self.multiplier * quantity

    def margin_required(self, price: float, quantity: int) -> float:
        """Initial margin required: notional * margin_rate."""
        return self.notional(price, quantity) * self.margin_rate

    def round_lot(self, quantity: int) -> int:
        """Round quantity down to nearest lot."""
        return (quantity // self.lot_size) * self.lot_size

    def valid_quantity(self, quantity: int) -> bool:
        """Check if quantity is a valid lot multiple."""
        return quantity > 0 and quantity % self.lot_size == 0

    def tick_round(self, price: float) -> float:
        """Round price to nearest tick."""
        if self.tick_size <= 0:
            return price
        return round(round(price / self.tick_size) * self.tick_size, 6)

    def commission(self, price: float, quantity: int, side: str = "buy") -> float:
        """Calculate commission for a trade.

        Stocks/ETFs: notional * commission_rate
        Futures: quantity * commission_per_lot (fixed per lot)
        Falls back to global commission_rate if per-instrument not set.
        """
        if self.commission_per_lot > 0:
            return quantity * self.commission_per_lot
        rate = self.commission_rate if self.commission_rate is not None else 0.0003
        return self.notional(price, quantity) * rate

    def stamp_tax(self, price: float, quantity: int, side: str = "buy") -> float:
        """Calculate stamp tax. Only applies to sell side for equities."""
        if side != "sell":
            return 0.0
        rate = self.stamp_tax_rate if self.stamp_tax_rate is not None else 0.001
        return self.notional(price, quantity) * rate

    @classmethod
    def stock(cls, symbol: str, exchange: str = "SSE") -> Instrument:
        """Factory for A-share stock (default parameters)."""
        # Determine price limit based on board
        price_limit = 0.10
        if symbol.startswith(("300", "301", "688")):
            price_limit = 0.20  # 创业板/科创板 20%

        return cls(
            symbol=symbol, asset_type=InstrumentType.STOCK,
            exchange=exchange, multiplier=1.0, tick_size=0.01,
            lot_size=100, margin_rate=1.0, t_plus=1,
            price_limit=price_limit,
        )

    @classmethod
    def etf(cls, symbol: str, exchange: str = "SSE") -> Instrument:
        """Factory for ETF (no stamp tax, smaller lot)."""
        return cls(
            symbol=symbol, asset_type=InstrumentType.ETF,
            exchange=exchange, multiplier=1.0, tick_size=0.001,
            lot_size=100, margin_rate=1.0, t_plus=1,
            stamp_tax_rate=0.0, price_limit=0.10,
        )

    @classmethod
    def index_future(cls, symbol: str, multiplier: float = 300,
                     margin_rate: float = 0.12) -> Instrument:
        """Factory for stock index futures (IF/IC/IM/IH)."""
        return cls(
            symbol=symbol, asset_type=InstrumentType.FUTURE,
            exchange="CFFEX", multiplier=multiplier, tick_size=0.2,
            lot_size=1, margin_rate=margin_rate,
            commission_rate=None, commission_per_lot=25.0,
            stamp_tax_rate=0.0, t_plus=0, price_limit=0.10,
        )

    @classmethod
    def commodity_future(cls, symbol: str, multiplier: float = 10,
                         tick_size: float = 1.0,
                         margin_rate: float = 0.10) -> Instrument:
        """Factory for commodity futures (cu, au, rb, etc.)."""
        return cls(
            symbol=symbol, asset_type=InstrumentType.FUTURE,
            exchange="SHFE", multiplier=multiplier, tick_size=tick_size,
            lot_size=1, margin_rate=margin_rate,
            commission_per_lot=3.0, stamp_tax_rate=0.0,
            t_plus=0, price_limit=0.06,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instrument:
        """Create from config dict."""
        at = data.get("asset_type", "stock")
        if isinstance(at, str):
            at = InstrumentType(at)
        return cls(
            symbol=data.get("symbol", ""),
            asset_type=at,
            exchange=data.get("exchange", "SSE"),
            multiplier=data.get("multiplier", 1.0),
            tick_size=data.get("tick_size", 0.01),
            lot_size=data.get("lot_size", 100),
            margin_rate=data.get("margin_rate", 1.0),
            commission_rate=data.get("commission_rate"),
            commission_per_lot=data.get("commission_per_lot", 0.0),
            stamp_tax_rate=data.get("stamp_tax_rate"),
            t_plus=data.get("t_plus", 1),
            price_limit=data.get("price_limit", 0.10),
            underlying=data.get("underlying", ""),
            expiry=data.get("expiry", ""),
            extra=data.get("extra", {}),
        )


class AssetUniverse:
    """Registry of tradeable instruments.

    Loads from config or manual registration. Provides lookup by symbol
    and filtering by asset type.
    """

    def __init__(self):
        self._instruments: dict[str, Instrument] = {}
        self._defaults: dict[InstrumentType, Instrument] = {}

    def add(self, instrument: Instrument) -> None:
        """Register an instrument."""
        self._instruments[instrument.symbol] = instrument

    def get(self, symbol: str) -> Instrument | None:
        """Look up instrument by symbol. Returns None if not found."""
        return self._instruments.get(symbol)

    def get_or_default(self, symbol: str) -> Instrument:
        """Look up instrument, falling back to stock defaults."""
        inst = self._instruments.get(symbol)
        if inst is not None:
            return inst
        return Instrument.stock(symbol)

    def filter_by_type(self, asset_type: InstrumentType) -> list[Instrument]:
        """Get all instruments of a given type."""
        return [i for i in self._instruments.values() if i.asset_type == asset_type]

    @property
    def symbols(self) -> list[str]:
        return list(self._instruments.keys())

    @property
    def count(self) -> int:
        return len(self._instruments)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._instruments

    def __len__(self) -> int:
        return len(self._instruments)

    @classmethod
    def from_config(cls, config: Any) -> AssetUniverse:
        """Load instruments from a config object or dict.

        Expected config structure (YAML):
            instruments:
              - symbol: "600519"
                asset_type: stock
                exchange: SSE
              - symbol: "IF2406"
                asset_type: future
                multiplier: 300
                lot_size: 1
                margin_rate: 0.12

        If no instruments section exists, returns an empty universe
        (backward compatible — get_or_default() returns stock defaults).
        """
        universe = cls()

        # Extract instruments list from config
        inst_list = None
        if hasattr(config, "instruments"):
            inst_list = getattr(config, "instruments", None)
        elif isinstance(config, dict):
            inst_list = config.get("instruments")

        if not inst_list:
            return universe

        for item in inst_list:
            if isinstance(item, dict):
                universe.add(Instrument.from_dict(item))
            elif isinstance(item, Instrument):
                universe.add(item)

        return universe

    @classmethod
    def default_a_share(cls) -> AssetUniverse:
        """Create a universe with sensible A-share defaults."""
        universe = cls()
        # No pre-registered instruments — get_or_default() handles it
        return universe
