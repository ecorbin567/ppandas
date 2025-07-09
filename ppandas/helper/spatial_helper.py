from shapely.geometry import Point, LineString, shape
import shapely.wkt
import numpy as np
import geopandas
from collections import defaultdict
import matplotlib.pyplot as plt

class SpatialHelper():
    @staticmethod
    # Input: List of strings, where each string is a geometric object value
    #       (MultPolygon,Polygon,Point) or a tuple of X and Y coordinates,
    #       of an old Bayes Net Location node
    # Ouput: Dictionary of strings, where key is string from input
    #        and value is a list of strings,
    #        mapping to new Bayes Net Location node
    def getMappingAsString(old_geometric_strings, new_geometric_strings):
        #Each new object should map to only 1 old object?
        mapping = defaultdict(list)
        for geo_old in old_geometric_strings:
            geo_old_obj = SpatialHelper.stringToGeoObject(geo_old)
            mapping[geo_old] = []
            for geo_new in new_geometric_strings:
                geo_new_obj = SpatialHelper.stringToGeoObject(geo_new)
                if geo_old_obj.geom_type == 'Point':
                    if geo_old_obj.within(geo_new_obj):
                        mapping[geo_old].append(geo_new)
                elif geo_old_obj.geom_type == 'Polygon' or \
                        geo_old_obj.geom_type == 'MultiPolygon':
                    if geo_old_obj.contains(geo_new_obj.buffer(-1e-10)):
                        mapping[geo_old].append(geo_new)
        numValues = 0
        OldRegionsNoMapping = 0
        for key, value in mapping.items():
            numValues += len(value)
            if len(value) == 0:
                OldRegionsNoMapping +=1
        return mapping

    @staticmethod
    def stringToGeoObject(tuple_xy_or_geo_string):
        if isinstance(tuple_xy_or_geo_string, tuple):
            # Convert any AtomicInterval elements to float midpoint
            coords = []
            for coord in tuple_xy_or_geo_string:
                if hasattr(coord, 'low') and hasattr(coord, 'high'):
                    coords.append((coord.low + coord.high) / 2)
                else:
                    coords.append(coord)
            return Point(coords)
        else:
            return shapely.wkt.loads(tuple_xy_or_geo_string)

    @staticmethod
    # output list of geo objects
    def getGeoObjectsFromString(string_geos):
        geos = []
        for geo in string_geos:
            geos.append(SpatialHelper.stringToGeoObject(geo))
        return geos

    @staticmethod
    # input:  Dictionary of strings where key is geometric object or
    #         tuple string and value is a list of strings,
    #         mapping to new Bayes Net Location node
    # output: dictionary where key are geometric object string
    #         and value is a list of geometric objects
    def convertMappingFromString(mapping):
        new_mapping = {}
        # geo_news can be a list of geometric objects
        for geo_old, geo_news in mapping.items():
            new_mapping[geo_old] = \
                SpatialHelper.getGeoObjectsFromString(geo_news)
        return new_mapping

    @staticmethod
    # output list of strings
    def convertGeoObjectsToString(geo_objects):
        string_geo_objects = []
        for geo in geo_objects:
            string_geo_objects.append(str(geo))
        return string_geo_objects

    @staticmethod
    # input: string, list of geometric objects, and old cpd value
    # Distribute cpd value for geo_old based on amount of overlap with
    # each of the geometric objects in geo_news
    def getUniformDistribution(geo_old, geo_news, value):
        # Convert string to geometric object
        geo_old_obj = SpatialHelper.stringToGeoObject(geo_old)
        size_portion = []
        for geo_new_obj in geo_news:
            geo_new_obj_buffer = geo_new_obj.buffer(0)
            if geo_old_obj.intersects(geo_new_obj):
                if geo_old_obj.area > 0:
                    size_portion.append(geo_old_obj.
                                        intersection(geo_new_obj_buffer).area
                                        / geo_old_obj.area)
                else:
                    size_portion.append(geo_old_obj.
                                        intersection(geo_new_obj).area)
            else:
                raise ValueError("{} does not overlap with {} "
                                 .format(geo_old, str(geo_new_obj)))
        size_portion = np.array(size_portion)
        return size_portion*value

    @staticmethod
    # input: list of strings of regions (Polygons or MultiPolygons)
    def computeNewRegions(ref_regions, sec_regions):
        ref_regions_obj = SpatialHelper.getGeoObjectsFromString(ref_regions)
        sec_regions_obj = SpatialHelper.getGeoObjectsFromString(sec_regions)
        
        # Intersection of all polgyons in both reference and secondary regions
        # Unique intersection of polygons 
        region_cross_product = []
        polys1 = geopandas.GeoSeries(ref_regions_obj)
        polys2 = geopandas.GeoSeries(sec_regions_obj)
        dfA = geopandas.GeoDataFrame({'geometry': polys1})
        dfB = geopandas.GeoDataFrame({'geometry': polys2})
        res_union = geopandas.overlay(dfA, dfB, how='intersection')
        res_union = res_union.geometry.explode()
        region_cross_product  = list(res_union.geometry.values)
        polys_cp = geopandas.GeoSeries(region_cross_product)
        polys_cp.boundary.plot()
        plt.axis('off')
        plt.savefig('wn_voronoi_comm_union.pdf')
        plt.show()

        non_overlapping = []
        for p in region_cross_product:
            overlaps = []
            for g in filter(lambda g: not g.equals(p), region_cross_product):
                overlaps.append(g.overlaps(p))
            if not any(overlaps):
                non_overlapping.append(p)
        
        return SpatialHelper.convertGeoObjectsToString(region_cross_product)