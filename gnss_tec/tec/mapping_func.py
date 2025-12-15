from typing import Protocol

import numpy as np
import polars as pl

from .constants import Re, TECConfig


class MappingFunctionProtocol(Protocol):
    def __call__(
        self,
        azimuth: pl.Expr,
        elevation: pl.Expr,
        rx_lat_deg: pl.Expr,
        rx_lon_deg: pl.Expr,
        config: TECConfig,
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """
        Calculate the mapping function and Ionospheric Pierce Point (IPP) latitude and
        longitude.

        Args:
            azimuth (pl.Expr): Satellite azimuth angle in degrees.
            elevation (pl.Expr): Satellite elevation angle in degrees.
            rx_lat_deg (pl.Expr): Receiver latitude in degrees.
            rx_lon_deg (pl.Expr): Receiver longitude in degrees.
            config (TECConfig): Configuration parameters.

        Returns:
            (pl.Expr, pl.Expr, pl.Expr): A tuple containing,
                - Mapping function (pl.Expr)
                - IPP latitude in degrees (pl.Expr)
                - IPP longitude in degrees (pl.Expr)
        """
        ...


def single_layer_model(
    azimuth: pl.Expr,
    elevation: pl.Expr,
    rx_lat_deg: pl.Expr,
    rx_lon_deg: pl.Expr,
    config: TECConfig,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    az = azimuth.radians()
    el = elevation.radians()
    rx_lat = rx_lat_deg.radians()
    rx_lon = rx_lon_deg.radians()

    # mapping function
    sin_beta = Re * el.cos() / (Re + config.ipp_height_m)
    mf = sin_beta.arcsin().cos().pow(-1)

    # IPP latitude and longitude, in radians
    psi = np.pi / 2 - el - sin_beta.arcsin()
    ipp_lat = (rx_lat.sin() * psi.cos() + rx_lat.cos() * psi.sin() * az.cos()).arcsin()
    ipp_lon = rx_lon + (psi.sin() * az.sin() / ipp_lat.cos()).arcsin()

    return mf, ipp_lat.degrees(), ipp_lon.degrees()


def modified_single_layer_model(
    azimuth: pl.Expr,
    elevation: pl.Expr,
    rx_lat_deg: pl.Expr,
    rx_lon_deg: pl.Expr,
    config: TECConfig,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    az = azimuth.radians()
    el = elevation.radians()
    rx_lat = rx_lat_deg.radians()
    rx_lon = rx_lon_deg.radians()

    # mapping function
    sin_beta = (
        Re * (np.pi / 2 - el).mul(config.alpha).sin() / (Re + config.mslm_height_m)
    )
    mf = sin_beta.arcsin().cos().pow(-1)

    # IPP latitude and longitude, in radians
    psi = np.pi / 2 - el - sin_beta.arcsin()
    ipp_lat = (rx_lat.sin() * psi.cos() + rx_lat.cos() * psi.sin() * az.cos()).arcsin()
    ipp_lon = rx_lon + (psi.sin() * az.sin() / ipp_lat.cos()).arcsin()

    return mf, ipp_lat.degrees(), ipp_lon.degrees()
