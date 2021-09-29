"""
Defines functions to load data from epygram / vortex.
"""
import numpy as np
import epygram
import usevortex as vtx


def load_data_from_grib(basis, term, vconf):
    """
    Loads the input data for the model from a grib.
    :param basis: datetime.datetime object indicating the basis for
        the requested validity.
    :param term: datetime.timedelta, term for the validity.
    :param vconf: Domain (indien, antilles, caledonie, polynesie)
    :return: A tuple (D, lats, longs) where:
    - D is the data, as an array of shape (2, H, W).
        The channels are FF10m and TA850,
        and are rotated by 90° clockwise to fit the model's expectations.
    - lats is a 1D array of shape (H,) giving the latitude at each row;
    - longs is a 1D array of shape (W,) giving the longitude at each column.
    """
    # SETUP
    epygram.init_env()
    XPID = "oper"
    cutoff = "production"
    '''
    vapp = "arpege"
    vconf = "4dvarfr"
    '''
    vapp = "arome"  # "arpege" "arome"
    members = None  # list(range(1,17,1))
    param = "ff10m"
    ########

    # Geometries
    Geometries = {}
    Geometries["antilles"] = "caraib0025"
    Geometries["indien"] = "indien0025"
    Geometries["caledonie"] = "ncaled0025"
    Geometries["polynesie"] = "polyn0025"
    Geometries["4dvarfr"] = "glob025"

    # GRIB ID
    # GRIB2 help : http://intra.cnrm.meteo.fr/gws/wtg/ --> Concept -->
    # enter name and copy dict to clipboard
    GribID = {}
    GribID["rr"] = {'parameterNumber': 65}  # RAIN GRIB2
    GribID["tpw850"] = {
        "discipline": 0,
        "parameterCategory": 0,
        "parameterNumber": 3,
        "tablesVersion": 15,
        "level": 850
    }
    GribID["ff10m"] = ({
        "discipline": 0,
        "parameterCategory": 2,
        "parameterNumber": 3,
        "scaleFactorOfFirstFixedSurface": 0,
        "scaledValueOfFirstFixedSurface": 10,
        "tablesVersion": 15,
        "typeOfFirstFixedSurface": 103
    }, {
        "discipline": 0,
        "parameterCategory": 2,
        "parameterNumber": 2,
        "scaledValueOfFirstFixedSurface": 10,
        "tablesVersion": 15,
        "typeOfFirstFixedSurface": 103
    })
    GribID["ta850"] = {
        'parameterCategory': 2,
        'parameterNumber': 10,
        'level': 850
    }

    # GRIB RETRIEVING
    # retrieves the term (hours) as an integer
    term = int(term.total_seconds() / 3600)
    resource = vtx.get_resources(experiment=XPID,
                                 date=basis,
                                 term=term,
                                 getmode='epygram',
                                 model=vapp,
                                 origin='hst',
                                 kind='gridpoint',
                                 block='forecast',
                                 cutoff=cutoff,
                                 vapp=vapp,
                                 vconf=vconf,
                                 geometry=Geometries[vconf],
                                 namespace='vortex.multi.fr',
                                 nativefmt='grib',
                                 members=members,
                                 shouldfly=True,
                                 uselocalcache=False)

    # FIELDS COMPUTATION FROM THE GRIB
    fieldU = resource[0].readfield(GribID[param][0])
    fieldV = resource[0].readfield(GribID[param][1])
    fieldVect = epygram.fields.make_vector_field(fieldU, fieldV)
    field = fieldVect.to_module()  # sqrt(U*U + V*V)

    # Reads the TA850 parameter
    field_ta850 = resource[0].readfield(GribID["ta850"])
    resource[0].close()

    # myGRIB.writefield(field)
    # Concatenate the ff10m and ta850 fields into a numpy image of
    # 2 channels
    ff10m_channel = field.getdata()
    ta850_channel = field_ta850.getdata()

    # Get the absolute value of the TA850 field since the model was trained
    # on northern hemisphere examples, where it is positive
    ta850_channel = np.abs(ta850_channel)

    data = np.swapaxes(np.stack([ff10m_channel, ta850_channel], axis=0), 1, 2)

    # Retrieves the domain's lonlat coords from the GRIB
    longs, lats = field_ta850.geometry.get_lonlat_grid()
    longs, lats = longs[0], lats[:, 0]

    # Reverses the latitudes as pixel 0 is actually the highest latitude
    lats = lats[::-1]

    return np.ascontiguousarray(data), lats, longs
