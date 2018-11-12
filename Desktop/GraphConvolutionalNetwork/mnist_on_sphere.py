from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import mnist_utils
from PIL import Image
import cv2
from Point import Point
import random
from graph_utils import *
from Graph import *

def draw_3d_points(x,y,z, color='b', size=1):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
    ax.scatter(x, y, z, s=size, c=color)
    plt.show()

def subsample_sphere(npoints):
    xi, yi, zi = sample_spherical(npoints)
    draw_3d_points(xi,yi,zi)
    return xi, yi, zi


#NON FUNZIONANTI
def sample_spherical(npoints, ndim=3):
    # https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
def inverse_mercator_projection(img, size=1, R = 1):

    #https://stackoverflow.com/questions/12732590/how-map-2d-grid-points-x-y-onto-sphere-as-3d-points-x-y-z/12734509#12734509
    X = []
    Y = []
    Z = []

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})

    for (x,y), pix in np.ndenumerate(img):
        longitude = x * R
        latitude = 2 * math.atan(math.exp(y / R)) - (math.pi / 2)

        x1 = R * math.sin(latitude) * math.cos(longitude)
        y1 = R * math.sin(latitude) * math.sin(longitude)
        z1 = R * math.cos(latitude)

        X.append(x1)
        Y.append(y1)
        Z.append(z1)

        ax.scatter(x1, y1, z1, s=20, c=str(pix/255))
    #ax.scatter(X, Y, Z, s=size, c='0')
    plt.show()

    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u, v = np.mgrid[0:np.pi:50j, 0:2 * np.pi:50j]

    strength = u
    norm = colors.Normalize(vmin=np.min(strength),
                            vmax=np.max(strength), clip=False)

    x = 10 * np.sin(u) * np.cos(v)
    y = 10 * np.sin(u) * np.sin(v)
    z = 10 * np.cos(u)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False,
                    facecolors=cm.coolwarm(norm(strength)))

    plt.show()
    '''
def mpl_to_plotly(cmap, pl_entries):
    h=1.0/(pl_entries-1)
    pl_colorscale=[]
    for k in range(pl_entries):
        C=map(np.uint8, np.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([round(k*h,2), 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale
def image_on_sphere(filename):

    #fn = get_sample_data(filename, asfileobj=False)
    #img = mpimg.imread(filename)
    img = cv2.imread(filename,1)

    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    ax = plt.gca(projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    #x = 1 * np.outer(np.cos(u), np.sin(v))
    #y = 1 * np.outer(np.sin(u), np.sin(v))
    #z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y,  np.sin(0.02*x)*np.sin(0.02*y), rstride=5, cstride=5, facecolors=img)

    plt.show()


    '''
    img = cv2.imread(filename, 1)
    #cv2.imshow("img",img)
    #cv2.waitKey()

    cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
    ax = gca(projection='3d')
    ax.plot_surface(x, y, np.sin(0.02*x)*np.sin(0.02*y), rstride=5, cstride=5, facecolors=img)
    show()
    '''
    '''
    img = cv2.imread("1.png")
    norm = normalized(img)
    cv2.imshow("img",img)
    cv2.waitKey()
    '''

    '''
    x = np.linspace(0,5,200)
    y = np.linspace(5,10,200)
    X, Y = np.meshgrid(x,y)
    z = (X+Y)/(2+np.cos(x)*np.sin(Y))

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    img = plt.imread("1.png")
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    print 'image shape', img.shape
    surfcolor = np.fliplr(img[200:400, 200:400])

    pl_grey = mpl_to_plotly(plt.get_cmap('gray'), 21)

    surf = Surface(x=x, y=y, z=z,
                   colorscale=pl_grey,
                   surfacecolor=surfcolor,
                   showscale=False
                   )

    noaxis = dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        ticks='',
        title='',
        zeroline=False)
    layout = Layout(
        title='Mapping an image onto a surface',
        font=Font(family='Balto'),
        width=800,
        height=800,
        scene=Scene(xaxis=XAxis(noaxis),
                    yaxis=YAxis(noaxis),
                    zaxis=ZAxis(noaxis),
                    aspectratio=dict(x=1,
                                     y=1,
                                     z=0.5
                                     ),
                    )
    )

    fig = Figure(data=[surf], layout=layout)
    py.sign_in('gsap91', 'I2FOtdLAIqyNSqYEyan9')
    py.iplot(fig, filename='mappingMNIST')
    '''
def equirectangolar_projection(digit):
    # Image Projection onto Sphere
    # https://en.wikipedia.org/wiki/Equirectangular_projection
    # Download the test image from the Wikipedia page!
    # FB36 - 20160731
    import math, random
    from PIL import Image
    imgxOutput = 768
    imgyOutput = 768
    pi2 = math.pi * 2

    # 3D Sphere Rotation Angles (arbitrary)
    xy = -pi2 * random.random()
    xz = -pi2 * random.random()
    yz = -pi2 * random.random()

    sxy = math.sin(xy)
    cxy = math.cos(xy)

    sxz = math.sin(xz)
    cxz = math.cos(xz)

    syz = math.sin(yz)
    cyz = math.cos(yz)

    imageInput = Image.open(get_random_mnist(digit)).convert('RGB')
    (imgxInput, imgyInput) = imageInput.size
    pixelsInput = imageInput.load()


    imgInput = cv2.imread(get_random_mnist(digit),0)
    imgOutput = np.zeros_like(imgInput)

    imageOutput = Image.new("RGB", (imgxOutput, imgyOutput),"white")
    pixelsOutput = imageOutput.load()

    # define a sphere behind the screen
    #centro sfera
    xc = (imgxOutput - 1.0) / 2
    yc = (imgyOutput - 1.0) / 2
    zc = min((imgxOutput - 1.0), (imgyOutput - 1.0)) / 2

    r = min((imgxOutput - 1.0), (imgyOutput - 1.0)) / 2

    # define eye point
    xo = (imgxOutput - 1.0) / 2
    yo = (imgyOutput - 1.0) / 2
    zo = -min((imgxOutput - 1.0), (imgyOutput - 1.0))

    xoc = xo - xc
    yoc = yo - yc
    zoc = zo - zc

    doc2 = xoc * xoc + yoc * yoc + zoc * zoc

    X = []
    Y = []
    Z = []
    flag = 0

    for yi in range(imgyOutput):
        for xi in range(imgxOutput):
            xio = xi - xo
            yio = yi - yo
            zio = 0.0 - zo

            dio = math.sqrt(xio * xio + yio * yio + zio * zio)

            xl = xio / dio
            yl = yio / dio
            zl = zio / dio

            dot = xl * xoc + yl * yoc + zl * zoc

            val = dot * dot - doc2 + r * r
            if val >= 0:  # if there is line-sphere intersection
                if val == 0:  # 1 intersection point
                    d = -dot
                else:  # 2 intersection points => choose the closest
                    d = min(-dot + math.sqrt(val), -dot - math.sqrt(val))
                    xd = xo + xl * d
                    yd = yo + yl * d
                    zd = zo + zl * d

                    x = (xd - xc) / r
                    y = (yd - yc) / r
                    z = (zd - zc) / r

                    x0 = x * cxy - y * sxy
                    y = x * sxy + y * cxy


                    x = x0  # xy-plane rotation
                    x0 = x * cxz - z * sxz
                    z = x * sxz + z * cxz

                    x = x0  # xz-plane rotation
                    y0 = y * cyz - z * syz
                    z = y * syz + z * cyz

                    y = y0  # yz-plane rotation

                    lng = (math.atan2(y, x) + pi2) % pi2
                    lat = math.acos(z)

                    X.append(x)
                    Y.append(y)
                    Z.append(z)


                    #proiezione
                    ix = int((imgxInput - 1) * lng / pi2 + 0.5)
                    iy = int((imgyInput - 1) * lat / math.pi + 0.5)

                    try:
                        pixelsOutput[xi, yi] = pixelsInput[ix, iy]
                        imgOutput[xi, yi] = imgInput[ix, iy]
                    except:
                        pass

    draw_3d_points(X, Y, Z)

    imageOutput.save(digit + ".png", "PNG")



#FUNZIONANTE (#http://milesbarnhart.com/portfolio/python/python-3d-satellite-orbital-trajectory-simulation/)
def latslons_to_XYZ(lats,lons,r):
    X = r * np.outer(np.cos(lons), np.cos(lats)).T
    Y = r * np.outer(np.sin(lons), np.cos(lats)).T
    Z = r * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    return X, Y, Z

def latlong_to_xyz(lat,long,r):
    x = r * np.cos(lat) * np.cos(long)
    y = r * np.cos(lat) * np.sin(long)
    z = r * np.sin(lat)
    return x, y, z


def plotSphere(img, radius=1.0, scale=1, stride=1):

    img = np.array(img.resize([int(d / scale) for d in img.size])) / 256.0
    img = mnist_utils.random_rotation(img)

    lats = np.linspace(-90, 90, img.shape[0])[::-1] * np.pi / 180
    lons = np.linspace(-180, 180, img.shape[1]) * np.pi / 180

    X, Y ,Z = latslons_to_XYZ(lats,lons,radius)

    latlongd = {}
    xyzd = {}

    for i in range(0,lats.shape[0]):
        for j in range(0,lons.shape[0]):
            x, y, z = latlong_to_xyz(lats[i],lons[j],radius)
            xyzd[(x,y,z)] = img[int(i/scale)][int(j/scale)]
            latlongd[(lats[i],lons[j])] = img[int(i/scale)][int(j/scale)]

    fig = plt.figure("Sphere projection (MNIST on sphere)")
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, alpha=1, facecolors=img)

    return X, Y, Z, lats, lons, xyzd, latlongd



def adj_points(p,lats,lons):
    lats = sorted(lats)
    lons = sorted(lons)
    lat_pos = lats.index(p.lat)
    long_pos = lons.index(p.long)


    if (long_pos - 1 < 0):
        long_a = lons[len(lons) - 1]
    else:
        long_a = lons[long_pos - 1]

    if (lat_pos - 1 < 0):
        lat_d = lats[len(lats) - 1]
    else:
        lat_d = lats[lat_pos - 1]

    if (lat_pos + 1 > len(lats) - 1):
        lat_b = lats[0]
    else:
        lat_b = lats[lat_pos + 1]
    if (long_pos + 1 > len(lons) - 1):
        long_c = lons[0]
    else:
        long_c = lons[long_pos + 1]

    a = Point(lats[lat_pos],long_a)
    b = Point(lat_b,lons[long_pos])
    c = Point(lats[lat_pos],long_c)
    d = Point(lat_d,lons[long_pos])

    return a, b, c, d



def mnist_on_sphere_to_graph(lats,lons,img):
    print ("(MNIST on sphere) Loading graph...")
    pos = {}
    n = 0
    lats = sorted(lats)
    lons = sorted(lons)

    nodes = []

    for i in range(0,len(lats)):
        for j in range(0,len(lons)):
            p = Point(lats[i],lons[j])
            pos[(p.lat,p.long)] = n
            nodes.append(float(img[i,j]))
            n+=1


    A = np.zeros(shape=(len(lats)*len(lons), len(lats)*len(lons)), dtype=np.uint8)


    #TODO: errata, matrice di adiacenza di una griglia!
    for i in range(0,len(lats)):
        for j in range(0,len(lons)):
            p = Point(lats[i],lons[j])
            a, b, c, d = adj_points(p, lats, lons)
            #a-p
            if (A[pos[(a.lat, a.long)]][pos[(p.lat, p.long)]] == 0):
                A[pos[(a.lat, a.long)]][pos[(p.lat, p.long)]] = 1
            if (A[pos[(p.lat, p.long)]][pos[(a.lat, a.long)]] == 0):
                A[pos[(p.lat, p.long)]][pos[(a.lat, a.long)]] = 1
            #b-p
            if (A[pos[(b.lat, b.long)]][pos[(p.lat, p.long)]] == 0):
                A[pos[(b.lat, b.long)]][pos[(p.lat, p.long)]] = 1
            if (A[pos[(p.lat, p.long)]][pos[(b.lat, b.long)]] == 0):
                A[pos[(p.lat, p.long)]][pos[(b.lat, b.long)]] = 1

            #c-p
            if (A[pos[(c.lat, c.long)]][pos[(p.lat, p.long)]] == 0):
                A[pos[(c.lat, c.long)]][pos[(p.lat, p.long)]] = 1
            if (A[pos[(p.lat, p.long)]][pos[(c.lat, c.long)]] == 0):
                A[pos[(p.lat, p.long)]][pos[(c.lat, c.long)]] = 1

            #d-p
            if (A[pos[(d.lat, d.long)]][pos[(p.lat, p.long)]] == 0):
                A[pos[(d.lat, d.long)]][pos[(p.lat, p.long)]] = 1
            if (A[pos[(p.lat, p.long)]][pos[(d.lat, d.long)]] == 0):
                A[pos[(p.lat, p.long)]][pos[(d.lat, d.long)]] = 1

    return A, nodes


def generate_graph(plot_data=False, test=False):

    digit = random.randint(0, 9)
    if digit < 0 or digit > 9:
        exit(1)

    chosen = mnist_utils.get_random_mnist(str(digit),test)

    img = Image.open(chosen).convert('RGB')
    img_gray = cv2.imread(chosen, 0)

    X, Y, Z, lats, lons, xyzd, latlongd = plotSphere(img, 1.0, 1)

    img_gray = np.divide(img_gray,255)

    A, X = mnist_on_sphere_to_graph(lats, lons, img_gray)
    A, X = add_fictional_node(A, X, 0)

    if plot_data:
        cv2.imshow(str(digit),img_gray)
        nx_G = data_to_networkx_graph(A,X)
        fig = plt.figure("Networkx graph (MNIST on sphere)")
        nx.draw(nx_G, with_labels=False, node_size=1.0, width=0.10)
        plt.show()

    if digit == 9:
        digit = 6

    G = Graph(A,X,digit)

    return G

def generate_dataset(n,test=False):
    graph_list = []
    for i in range(0, n):
        G = generate_graph(test=test)
        graph_list.append(G)

        print(' ' + str(i+1) + '/' + str(n))
    print("(MNIST on sphere) Dataset created.")
    return graph_list
