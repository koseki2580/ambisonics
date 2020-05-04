import numpy as np 
import function as fu


def Plane_wave(freq,cv, pos_s,vec_dn, pos_r):
    # freq: frequency
    # cv: sound speed
    # pos_s: source position (Not used in this function)
    # vec_dn: plane wave direction vector
    # pos_r: receiver position
    omega = 2*np.pi*freq
    k = omega/cv
    vec_dn_unit = np.array([0]*3,dtype = 'float')
    vec_dn_unit[0] = vec_dn[0] / np.linalg.norm(vec_dn, ord=2)
    vec_dn_unit[1] = vec_dn[1] / np.linalg.norm(vec_dn, ord=2)
    vec_dn_unit[2] = vec_dn[2] / np.linalg.norm(vec_dn, ord=2)
    azi,ele,_ = fu.cart2sph(-vec_dn_unit[0],-vec_dn_unit[1],-vec_dn_unit[2])
    ele_1 = np.pi / 2 - ele
    kx = k * np.cos(azi) * np.sin(ele_1)
    ky = k * np.sin(azi) * np.sin(ele_1)
    kz = k * np.cos(ele_1)
    x = pos_r[0] - pos_s[0]
    y = pos_r[1] - pos_s[1]
    z = pos_r[2] - pos_s[2]
    ret = np.exp(1j*(kx*x + ky*y + kz*z))
    return ret

def green(freq, cv, vec1, vec2):
    vec = np.array([0]*3,dtype = 'float')
    vec[0] = vec1[0] - vec2[0] 
    vec[1] = vec1[1] - vec2[1]
    vec[2] = vec1[2] - vec2[2]
    rr = np.linalg.norm(vec, ord=2)
    omega = 2 * np.pi * freq
    k = omega / cv
    jkr = 1j * k * rr
    ret = np.exp(jkr)/(4 * np.pi * rr)
    return ret

def green2(alpha, freq, cv, vecrs, vecr):
    # alpha:directivity coefficient 0~1 
    #                               0   :dipole 
    #                               0.5 :cardiod 
    #                               1   :monopole
    # CV: sound speed
    # freq: frequency [Hz] (can be vector)
    # vecrs: sound source position
    # vecr: receiver position
    vecrd = np.array([0]*3,dtype = 'float')
    vecrd[0] = vecrs[0] - vecr[0]
    vecrd[1] = vecrs[1] - vecr[1]
    vecrd[2] = vecrs[2] - vecr[2]
    rd = np.linalg.norm(vecrd, ord=2)
    omega = 2 * np.pi * freq
    k = omega/cv
    cosgamma = np.dot(vecrd,-vecrs) / (np.linalg.norm(vecrd, ord=2) * np.linalg.norm(vecrs, ord=2))
    ret = green(freq, cv, vecrs, vecr) * (alpha - (1 - alpha)*(1 + (1j/(k*rd)))*cosgamma)
    return ret
    

