import numpy as np
from scipy import ndimage, optimize
import pdb 
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import multiprocessing
import datetime

####################################################
def findMaxRect(data):
   
    '''http://stackoverflow.com/a/30418912/5008845'''

    nrows,ncols = data.shape
    w = np.zeros(dtype=int, shape=data.shape)
    h = np.zeros(dtype=int, shape=data.shape)
    skip = 1
    area_max = (0, [])
   

    for r in range(nrows):
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r-1][c]+1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c-1]+1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                area = (dh+1)*minw
                if area > area_max[0]:
                    area_max = (area, [(r-dh, c-minw+1, r, c)])

    return area_max


########################################################################
def residual(angle,data,flag_bfgs=False):
   
    if flag_bfgs:
        angle = 360*angle

    nx,ny = data.shape
    M = cv2.getRotationMatrix2D(((nx-1)/2,(ny-1)/2),angle,1)
    RotData = cv2.warpAffine(data,M,(nx,ny),flags=cv2.INTER_NEAREST,borderValue=1)
    rectangle = findMaxRect(RotData)
   
    return 1./rectangle[0]


########################################################################
def residual_star(args):
    return residual(*args)
    

########################################################################
def get_rectangle_coord(angle,data,flag_out=None):
    nx,ny = data.shape
    M = cv2.getRotationMatrix2D(((nx-1)/2,(ny-1)/2),angle,1)
    RotData = cv2.warpAffine(data,M,(nx,ny),flags=cv2.INTER_NEAREST,borderValue=1)
    rectangle = findMaxRect(RotData)
    
    if flag_out:
        return rectangle[1][0], M, RotData
    else:
        return rectangle[1][0], M


########################################################################
def findRotMaxRect(data_in,flag_opt=False,flag_parallel = True, nbre_angle=10,flag_out=None,fag_enlarge_img=True):
    
    nx_in,ny_in = data_in.shape
    rotation_angle = np.linspace(0,360,nbre_angle+1)[:-1]

    if fag_enlarge_img:
        data = np.zeros([2*data_in.shape[0]+1,2*data_in.shape[1]+1]) + 1
        nx,ny = data.shape
        data[nx/2-nx_in/2:nx/2+nx_in/2,ny/2-ny_in/2:ny/2+ny_in/2] = data_in
    else:
        data = np.copy(data_in)
        nx,ny = data.shape
    
    if flag_opt:
        myranges_brute = ([(0.,360.),])
        myranges_bfg = ([(0.,1.),])
        coeff0 = np.array([0.,])
        coeff1  = optimize.brute(residual, myranges_brute, args=(data,), Ns=4, finish=None)
        popt = optimize.fmin(residual, coeff1, args=(data,), xtol=5., ftol=1.e-5,disp=False)
        angle_selected = popt[0]

    else:
        args_here=[]
        for angle in rotation_angle:
            args_here.append([angle,data])
        
        if flag_parallel: 
   
            # set up a pool to run the parallel processing
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation  

            results = pool.map(residual_star, args_here)
            
            pool.close()
            pool.join()
            
       
        else:
            results = []
            for arg in args_here:
                results.append(residual_star(arg))
                
        argmin = np.array(results).argmin()
        angle_selected = args_here[argmin][0]
    #rectangle, M_rect_max, RotData  = get_rectangle_coord(angle_selected,data,flag_out=True)
    rectangle, M_rect_max  = get_rectangle_coord(angle_selected,data)

    #invert rectangle 
    M_invert = cv2.invertAffineTransform(M_rect_max)
    rect_coord = [rectangle[:2], [rectangle[0],rectangle[3]] , 
                  rectangle[2:], [rectangle[2],rectangle[1]] ]
   
    #ax = plt.subplot(111)
    #ax.imshow(RotData.T,origin='lower',interpolation='nearest')
    #patch = patches.Polygon(rect_coord, edgecolor='k', facecolor='None', linewidth=2)
    #ax.add_patch(patch)
    #plt.show()

    rect_coord_ori = []
    for coord in rect_coord:
        rect_coord_ori.append(np.dot(M_invert,[coord[0],(ny-1)-coord[1],1]))

    #transform to numpy coord of input image
    coord_out = []
    for coord in rect_coord_ori:
        coord_out.append(    [round(coord[0],0)-(nx/2-nx_in/2),round((ny-1)-coord[1],0)-(ny/2-ny_in/2)])
    
    coord_out_rot = []
    for coord in rect_coord:
        coord_out_rot.append( [ (round(coord[0],0))-(nx/2-nx_in/2), round(coord[1],0)-(ny/2-ny_in/2)] )

    if flag_out is None:
        return coord_out
    elif flag_out == 'rotation':
        return coord_out, angle_selected,coord_out_rot
    else:
        print 'bad def in findRotMaxRect input. stop'
        pdb.set_trace()

#######################################
if __name__ == '__main__':
#######################################

    scale_factor = 3

    #read image
    a = cv2.cvtColor(cv2.imread('3VcIL.png'),cv2.COLOR_BGR2GRAY)
    idx_in  = np.where(a==255) 
    idx_out = np.where(a==0) 
    aa = np.ones([300,300])
    aa[idx_in]  = 0
  
    #apply scale factor
    bb = cv2.resize(aa,(aa.shape[0]/scale_factor,aa.shape[1]/scale_factor),interpolation=0)
     
    # set the input data with an odd number of point in each dimension to make rotation easier
    nx,ny = bb.shape
    nx_extra = 0; ny_extra = 0   
    if nx%2==0:
        nx+=1
        nx_extra = 1
    if ny%2==0:
        ny+=1
        ny_extra = 1
    bb2 = np.ones([bb.shape[0]+nx_extra,bb.shape[1]+ny_extra])
    bb2[:-nx_extra,:-ny_extra] = bb
    nx,ny = bb2.shape

    #get coordinate of biggest rectangle
    time_start = datetime.datetime.now()
    rect_coord_ori, angle, coord_out_rot= findRotMaxRect(bb2, flag_opt=True, flag_parallel =False,  nbre_angle=4, flag_out='rotation',fag_enlarge_img=False)
    coord_aa = []
    for coord in rect_coord_ori:
        coord_aa.append([scale_factor*coord[0],scale_factor*coord[1]])
    
    '''
    M = cv2.getRotationMatrix2D(((aa.shape[0]-1)/2,(aa.shape[1]-1)/2),angle,1)
    RotData = cv2.warpAffine(aa,M,aa.shape,flags=cv2.INTER_NEAREST,borderValue=1)
    bb = cv2.resize(RotData,(RotData.shape[0]/scale_factor,RotData.shape[1]/scale_factor),interpolation=0)
    rect_coord_ori_rot = findRotMaxRect(bb, flag_opt=True, flag_parallel =False,  nbre_angle=1)
    coord_aa_rot = []
    for coord in rect_coord_ori_rot:
        coord_aa_rot.append([scale_factor*coord[0],scale_factor*coord[1]])
    '''

    M = cv2.getRotationMatrix2D(((bb2.shape[0]-1)/2,(bb2.shape[1]-1)/2),angle,1)
    RotData = cv2.warpAffine(bb2,M,bb2.shape,flags=cv2.INTER_NEAREST,borderValue=1)
    

    print 'time elapsed =', (datetime.datetime.now()-time_start).total_seconds()

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(aa.T,origin='lower',interpolation='nearest')
    patch = patches.Polygon(coord_aa, edgecolor='green', facecolor='None', linewidth=2)
    ax.add_patch(patch)
    plt.show()
    
    ax = plt.subplot(111)
    ax.imshow(RotData.T,origin='lower',interpolation='nearest')
    patch = patches.Polygon(coord_out_rot, edgecolor='green', facecolor='None', linewidth=2)
    ax.add_patch(patch)
    plt.show()



