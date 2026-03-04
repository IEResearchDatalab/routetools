from collections.abc import Iterable

import h3.api.basic_int as h3
from h3 import Polygon as H3Polygon
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
            f"The ocean geometry is not valid. Please check the geojson file.{polygon}"
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


def get_h3_cells(polygons: list[dict], res: int = 5) -> set[int]:
    """
    Get the h3 cells from a list of polygons.

    Parameters
    ----------
    polygons : List[dict]
        A list of polygons with the coordinates of the geometry.
    res : int, optional
        The resolution of the h3 cells, by default 5.

    Returns
    -------
    set[int]
        A set of h3 cells.
    """
    cells = set()

    for polygon in polygons:
        exterior = [(lat, lon) for lon, lat in polygon.exterior.coords]
        interior = [
            [(lat, lon) for lon, lat in interior.coords[:]]
            for interior in polygon.interiors
        ]
        polygon_h3 = H3Polygon(exterior, *interior)
        cells_polygon = h3.polygon_to_cells(polygon_h3, res)

        # TODO: compact_cells is not working due to a bug from the library.
        # Once it is fixed, we can use it to reduce the number of cells
        # cells_compacted = h3.compact_cells(cells_polygon)

        cells.update(cells_polygon)

    return cells


def remove_border_cells(cells: set[int], num_cells: int = 1) -> set[int]:
    """
    Remove the cells that are next to land.

    Parameters
    ----------
    cells : set[int]
        A set of h3 cells.
    num_cells : int, optional
        Distance to border in cells, by default 1

    Returns
    -------
    set[int]
        A set of h3 cells.
    """
    updated_cells = set()

    for cell in cells:
        neighbours = h3.grid_disk(cell, num_cells)

        for n in neighbours:
            if n not in cells:
                # When a neighbor is not in the cell set, means that neighbor is land
                # We then consider this node as land and not
                # included it in the new cell set
                break
        else:
            updated_cells.add(cell)

    return updated_cells


def multipolygon_to_h3_cells(
    multipolygon: Polygon | MultiPolygon,
    res: int = 5,
    land_dilation: int = 1,
) -> set[int]:
    """
    Get the h3 cells from a geojson file.

    Parameters
    ----------
    multipolygon : Union[Polygon, MultiPolygon]
        A shapely polygon or multipolygon.
    res : int, optional
        The resolution of the h3 cells, by default 5.
    land_dilation : int, optional
        Distance to land from cells border, by default 1

    Returns
    -------
    set[int]
        A set of h3 cells.
    """
    polygons = []

    if isinstance(multipolygon, Polygon):
        polygons.append(multipolygon)
    else:
        # If it is a multipolygon, we need to get only the polygons from it
        for geom in multipolygon.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)

    cells = get_h3_cells(polygons, res=res)

    if land_dilation > 0:
        cells = remove_border_cells(cells, land_dilation)

    return cells
