import green_function as gr
import  function as fu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

#初期条件
ambi_order = 7              #打ち切り次数
freq = 550                  #周波数
pos_s = np.array([2,0,0])   #一次音源の座標x,y,z
vec_dn = np.array([1,0,0])  #平面波の向き
cv = 344                    #音速
r_sp = 1.5                  #スピーカーを配置する半径
r_mic = 0.05                #マイクロホンの半径
#グラフ条件
Plot_range = 2.5
space = 0.02

#計算
lambda_ = cv/freq
omega = 2*np.pi*freq
k = omega/cv


#スピーカー配置を決定
num_sp = (ambi_order+1)**2
xyz = fu.EquiDistSphere((ambi_order+1)**2)
xyz = np.array([xyz]).reshape(num_sp,3)

pos_sp = xyz * r_sp

azi_sp, ele_sp, rr_sp = fu.cart2sph(pos_sp[:,0],pos_sp[:,1],pos_sp[:,2])
ele_sp1 = np.pi /2 - ele_sp
#マイク配置を決定
num_mic = (ambi_order+1)**2
pos_mic = xyz * r_mic
azi_mic, ele_mic, rr_mic = fu.cart2sph(pos_mic[:,0],pos_mic[:,1],pos_mic[:,2])
ele_mic1 = np.pi /2 - ele_mic
mic_sig = np.array([])
for i in range(num_mic):
    mic_sig_ = gr.Plane_wave(freq,cv,pos_s,vec_dn,pos_mic[i])
    mic_sig = np.append(mic_sig, mic_sig_)
#エンコード
ambi_signal = fu.encode(ambi_order,azi_mic,ele_mic1,k,r_mic,mic_sig)

#デコード
spkr_output = fu.decode(ambi_order,azi_sp,ele_sp1,r_sp,lambda_,ambi_signal)

#所望音場(一次音源)と再現音場を求める
X = np.arange(-Plot_range,Plot_range+space,space)
Y = np.arange(-Plot_range,Plot_range+space,space)
XX,YY = np.meshgrid(X,Y)
P_org = np.zeros((X.size,Y.size), dtype = np.complex)
P_HOA = np.zeros((X.size,Y.size), dtype = np.complex)
for j in range(X.size):
    for i in range(Y.size):
        pos_r = np.array([X[j],Y[i],0])
        P_org[i,j] = gr.Plane_wave(freq,cv,pos_s,vec_dn,pos_r)
        for l in range(num_sp):
            G_transfer = gr.Plane_wave(freq,cv,pos_sp[l,:],-pos_sp[l,:],pos_r)
            P_HOA[i,j] += G_transfer *  spkr_output[l]

NE = np.zeros((X.size,Y.size), dtype = np.complex)
for i in range(X.size):
    for j in range(Y.size):
        NE[i,j] = 10*np.log10(((np.abs(P_HOA[i,j]-P_org[i,j]))**2)/((np.abs(P_org[i,j]))**2))

#スイートスポット
rad = (ambi_order*cv)/(2*np.pi*freq)

#グラフ
#所望音場
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False)
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b',fill=False)
plt.figure()
ax = plt.axes()
im = ax.imshow(np.real(P_org), interpolation='gaussian',cmap=cm.jet,origin='lower', 
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=1, vmin=-1)
ax.grid(False)
ax.add_patch(sweet_spot)
ax.add_patch(r_sp_)
plt.axis('scaled')
ax.set_aspect('equal')
plt.colorbar(im)
plt.savefig('original_sound.pdf',bbox_inches="tight",dpi = 64,
            facecolor = "lightgray", tight_layout = True) 
plt.show()

#再現音場
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False)
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b',fill=False)
plt.figure()
ax2 = plt.axes()
im2 = ax2.imshow(np.real(P_HOA), interpolation='gaussian',cmap=cm.jet,origin='lower', 
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=1, vmin=-1)
ax2.grid(False)
ax2.add_patch(sweet_spot)
ax2.add_patch(r_sp_)
plt.axis('scaled')
ax2.set_aspect('equal')
plt.colorbar(im2)
plt.savefig('reproduced_sound.pdf',bbox_inches="tight",dpi = 64,
            facecolor = "lightgray", tight_layout = True)
plt.show()

#正規化誤差
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False)
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b',fill=False)
plt.figure()
ax3 = plt.axes()
im3 = ax3.imshow(np.real(NE), interpolation='gaussian',cmap=cm.pink_r,origin='lower', 
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=0, vmin=-60)
ax3.grid(False)
ax3.add_patch(sweet_spot)
ax3.add_patch(r_sp_)
plt.axis('scaled')
ax3.set_aspect('equal')
plt.colorbar(im3)
plt.savefig('normalization_error.pdf',bbox_inches="tight",dpi = 64,
            facecolor = "lightgray", tight_layout = True)
plt.show()
