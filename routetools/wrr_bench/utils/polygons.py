from collections.abc import Iterable

from shapely.geometry import MultiPolygon, Polygon


def invert_polygon(
    polygon: Polygon | MultiPolygon,
    bounding_box: Iterable[float],
) -> Polygon | MultiPolygon:
    """
    Invert a Polygon or MultiPolygon with a bounding box.

    Parameters
    ----------
    polygon : Union[Polygon, MultiPolygon]
        The input geometry.
    bounding_box : Iterable[float]
        The bounding box as a list of four floats: [min_lat, min_lon, max_lat, max_lon].

    Returns
    -------
    Union[Polygon, MultiPolygon]
        The inverted geometry.
    """
    bounding_polygon = Polygon(
        [
            (bounding_box[1], bounding_box[0]),
            (bounding_box[3], bounding_box[0]),
            (bounding_box[3], bounding_box[2]),
            (bounding_box[1], bounding_box[2]),
        ]
    )
    try:
        polygon = bounding_polygon.difference(polygon)
    except Exception as e:
        raise Exception(
            "The ocean geometry is not valid. Please check the geojson file."
            f"{polygon}"
        ) from e

    return polygon


def crop_polygon(
    multipolygon: MultiPolygon, bounding_box: Iterable[float]
) -> MultiPolygon:
    """
    Crop a MultiPolygon with a bounding box.

    Parameters
    ----------
    multipolygon : MultiPolygon
        The input geometries as a MultiPolygon.
    bounding_box : Iterable[float]
        The bounding box as a list of four floats: [min_lat, min_lon, max_lat, max_lon].

    Returns
    -------
    MultiPolygon
        The cropped MultiPolygon.
    """
    # Crear el polígono de la bounding box
    bounding_polygon = Polygon(
        [
            (bounding_box[1], bounding_box[0]),
            (bounding_box[3], bounding_box[0]),
            (bounding_box[3], bounding_box[2]),
            (bounding_box[1], bounding_box[2]),
        ]
    )

    return multipolygon.intersection(bounding_polygon)


def relative_to_latlon(
    y: float, x: float, height: int, width: int, bounding_box: Iterable[float]
) -> Iterable[float]:
    """
    Transform the relative coordinates of a pixel to latitude and longitude.

    Parameters
    ----------
    y : float
        The relative y coordinate.
    x : float
        The relative x coordinate.
    height : int
        The height of the image.
    width : int
        The width of the image.
    bounding_box : Iterable[float]
        The bounding box as a list of four floats: [min_lat, min_lon, max_lat, max_lon].

    Returns
    -------
    Iterable[float]
        The latitude and longitude of the pixel.
    """
    lat_min, lon_min, lat_max, lon_max = bounding_box
    lat = lat_min + (lat_max - lat_min) * (y / height)
    lon = lon_min + (lon_max - lon_min) * (x / width)

    return lon, lat
