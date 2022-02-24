import numpy as np
import ROOT as r

data = np.loadtxt("/lfs/l2/hawc/data/maps/take4-dataset-kelly/take4-mc-corrected/eff-area.txt")

print data

nrows = data.shape[0]
eband_num = 1
sel_eband = np.arange(0 + eband_num, nrows + eband_num,4)
eband_data = data[sel_eband,:]

cos_th = np.cos(0.5*(np.deg2rad(eband_data[:,2]) + np.deg2rad(eband_data[:,3])))
cos_th = np.flip(cos_th,0)
eff_area = np.flip(eband_data[:,4],0)


print type(eff_area), type(cos_th)
print eband_data

graph = r.TGraph(cos_th.size,np.array(cos_th),np.array(eff_area))
graph.SetMarkerStyle(20)
#func = r.TF1("func","[0]*exp(-0.5*((x-[1])/[2])*((x-[1])/[2]))",cos_th[0],cos_th[-1])
func = r.TF1("func","[0] + [1]*x + [2]*x*x",cos_th[0],cos_th[-1])
func.SetParameter(0,np.mean(eff_area))
func.SetParameter(1,np.mean(eff_area)*0.5)
func.SetParameter(2,np.mean(eff_area)*0.1)

graph.Fit(func)


can = r.TCanvas("can","",600,600)
graph.Draw("ap")
can.SaveAs("test.pdf")

phi = (2*np.pi/(23*3600 + 56*60 + 4.1))*np.linspace(0,23*3600 + 56*60 + 4.1,23*3600 + 56*60 + 4.1) # per sec phi
delta = np.pi*0.5 - np.deg2rad(36.8)
lam =  np.pi*0.5 - np.deg2rad(19.0)

tr_cos_th = np.cos(phi)*np.sin(delta)*np.sin(lam) + np.cos(delta)*np.cos(lam)
tr_cos_th = tr_cos_th[tr_cos_th > 0.70]
print np.rad2deg(np.arccos(np.amax(tr_cos_th)))
tr_eff_area  = []

for ct in tr_cos_th:
    tr_eff_area.append(func.Eval(ct))
    
tr_eff_area = np.array(tr_eff_area)
tr_cos_th = np.rad2deg(np.arccos(tr_cos_th)) #in deg
tr_cos_th = (24*3600/360)*(tr_cos_th) #in sec

gr = r.TGraph(tr_cos_th.size,tr_cos_th ,tr_eff_area)
f = r.TF1("f","[0] + [1]*x + [2]*x*x",np.amin(tr_cos_th),np.amax(tr_cos_th))
gr.Fit(f)
final_eff_area = 2*f.Integral(np.amin(tr_cos_th),np.amax(tr_cos_th))

print final_eff_area
gr.SetMarkerStyle(20)
c = r.TCanvas("c","",600,600)
gr.Draw("aep")

c.SaveAs("test2.pdf")
