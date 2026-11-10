"""Offline observed-pixel geometry. Never imported by the training path.

Normalized linear triangulation follows COLMAP 3.11.1 triangulation.cc,
https://github.com/colmap/colmap/blob/3.11.1/src/colmap/geometry/triangulation.cc
See docs/research/observed-correspondence-launch.md for provenance/license.
"""
import numpy as np


def host_to_normalized(xy, shape):
    h,w=shape
    return 2*(np.asarray(xy)+.5)/np.array([w,h])-1


def normalized_to_host(xy, shape):
    h,w=shape
    return (np.asarray(xy)+1)*np.array([w,h])/2-.5


def inside(xy, shape):
    h,w=shape
    return np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)


def _sample(field, normalized):
    """Donor BHWC tensors -> fixed pixels, without padded border evidence."""
    import torch
    from torch.nn import functional as F
    if field.ndim<4 or field.shape[0]!=1:
        raise ValueError('expected single-pair BHWC donor field')
    trailing=tuple(field.shape[3:]);h,w=field.shape[1:3]
    pos=normalized_to_host(normalized,(h,w))
    footprint=inside(pos,(h,w))
    grid=torch.as_tensor(np.nan_to_num(normalized,nan=2.,posinf=2.,neginf=-2.),
        dtype=torch.float32,device=field.device)[None,None]
    flat=field.reshape(1,h,w,-1).permute(0,3,1,2).float()
    result=F.grid_sample(flat,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    result=result[0,:,0].T.detach().cpu().numpy().reshape((-1,*trailing)).copy()
    valid=footprint&np.isfinite(result.reshape(len(result),-1)).all(-1)
    result[~valid]=np.nan
    return result,valid,footprint


def query_matches(predictions, reference_xy, reference_shape, source_shape):
    """Use unmodified match() output, not donor stochastic sample()/KDE."""
    ref_norm=host_to_normalized(reference_xy,reference_shape)
    warp,finite,footprint=_sample(predictions['warp_AB'],ref_norm)
    source_xy=normalized_to_host(warp,source_shape)
    forward=finite&inside(source_xy,source_shape)&inside(reference_xy,reference_shape)
    backward,bfinite,bfootprint=_sample(predictions['warp_BA'],warp)
    back_xy=normalized_to_host(backward,reference_shape)
    cycle_valid=forward&bfinite
    cycle=np.linalg.norm(back_xy-reference_xy,axis=-1)
    cycle[~cycle_valid]=np.nan
    result=dict(reference_xy=np.asarray(reference_xy),source_xy=source_xy,backward_xy=back_xy,
        forward_valid=forward,forward_footprint_valid=footprint,
        source_in_bounds=inside(source_xy,source_shape),cycle_valid=cycle_valid,
        reverse_footprint_valid=bfootprint,backward_in_bounds=inside(back_xy,reference_shape),cycle_px=cycle)
    for key in ('overlap','precision'):
        f,_,_=_sample(predictions[key+'_AB'],ref_norm)
        b,_,_=_sample(predictions[key+'_BA'],warp)
        result[key+'_forward']=f;result[key+'_reverse']=b
    return result


def relative_pose(reference_transform, source_transform):
    """Saved row-vector world->camera transforms to source column pose."""
    ref=np.asarray(reference_transform,dtype=np.float64)
    src=np.asarray(source_transform,dtype=np.float64)
    r=src[:3,:3].T@ref[:3,:3]
    t=src[3,:3]-r@ref[3,:3]
    return r,t


def rays(xy,k):
    q=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(k).T
    return q/q[:,2:]


def project_depth(xy,depth,ref_k,ref_tr,src_k,src_tr,src_shape,ref_near,src_near):
    r,t=relative_pose(ref_tr,src_tr)
    points=rays(xy,ref_k)*np.asarray(depth)[:,None]
    source=points@r.T+t
    with np.errstate(divide='ignore',invalid='ignore'):
        pix=source@src_k.T;out=pix[:,:2]/pix[:,2:]
    valid=np.isfinite(depth)&(depth>ref_near)&np.isfinite(source).all(-1)&(source[:,2]>src_near)&inside(out,src_shape)
    return out,valid


def compare_candidates(costs, valid):
    """N x [current,DA3,MVS,spread] x supports; strictly shared supports."""
    costs=np.asarray(costs);valid=np.asarray(valid)&np.isfinite(costs)
    if costs.shape!=valid.shape or costs.ndim!=3 or costs.shape[1]!=4:
        raise ValueError('expected N x 4 candidates x supports')
    result={}
    for count in (3,4):
        common=valid[:,:count].all(1);n=common.sum(1)
        total=np.where(common[:,None],costs[:,:count],0).sum(2)
        means=np.full((len(costs),count),np.nan)
        means[n>0]=total[n>0]/n[n>0,None]
        result[f'common{count}_support_mask']=common
        result[f'common{count}_support_count']=n
        result[f'common{count}_means']=means
    m=result['common3_means']
    result['mvs_selected']=(m[:,2]<m[:,0])&(m[:,2]<m[:,1])
    m=result['common4_means']
    result['mvs_selected_common4']=(m[:,2]<m[:,0])&(m[:,2]<m[:,1])
    return result


def _linear_points(observations, projections):
    """Batch COLMAP normalized two-view SVD / multiview eigen operator."""
    if len(projections)==2:
        rows=[]
        for xy,p in zip(observations,projections):
            rows.extend([xy[:,0,None]*p[2]-p[0],xy[:,1,None]*p[2]-p[1]])
        a=np.stack(rows,axis=1)
        _,values,v=np.linalg.svd(a)
        homogeneous=v[:,-1]
        rank_ok=values[:,-2]>np.finfo(float).eps*4*values[:,0]
    else:
        normal=np.zeros((len(observations[0]),4,4))
        for xy,p in zip(observations,projections):
            point=np.c_[xy,np.ones(len(xy))];point/=np.linalg.norm(point,axis=-1,keepdims=True)
            term=p[None]-point[:,:,None]*(point@p)[:,None,:]
            normal+=term.transpose(0,2,1)@term
        values,v=np.linalg.eigh(normal)
        homogeneous=v[:,:,0]
        rank_ok=values[:,1]>np.finfo(float).eps*4*np.maximum(values[:,-1],1.)
    good=rank_ok&np.isfinite(homogeneous).all(-1)&(np.abs(homogeneous[:,3])>1e-12)
    points=np.full((len(homogeneous),3),np.nan)
    points[good]=homogeneous[good,:3]/homogeneous[good,3:]
    return points,good


def triangulate_evidence(reference_xy,source_xy,forward_valid,ref_k,ref_tr,src_ks,src_trs,
                         src_shapes,ref_near,src_nears):
    """Pair0, pair1, joint. All depth targets use original reference rays."""
    xy=np.asarray(reference_xy,dtype=float);qs=np.asarray(source_xy,dtype=float)
    n=len(xy);rr=rays(xy,ref_k)
    ps=[np.c_[np.eye(3),np.zeros(3)]]
    for tr in src_trs:
        r,t=relative_pose(ref_tr,tr);ps.append(np.c_[r,t])
    cameras=[ref_k,*src_ks];all_near=[ref_near,*src_nears]
    observations=[xy,qs[:,0],qs[:,1]]
    def solve(used,obs):
        okay=np.ones(n,bool)
        for s in used[1:]:okay&=forward_valid[:,s-1]&inside(obs[s],src_shapes[s-1])
        for s in used:okay&=np.isfinite(obs[s]).all(-1)
        points=np.full((n,3),np.nan)
        ids=np.flatnonzero(okay)
        if len(ids):
            coords=[rays(obs[s][ids],cameras[s])[:,:2] for s in used]
            x,valid=_linear_points(coords,[ps[s] for s in used])
            points[ids]=x;okay[ids]&=valid
        for s in used:
            camera_points=points@ps[s][:,:3].T+ps[s][:,3]
            okay&=np.isfinite(camera_points).all(-1)&(camera_points[:,2]>all_near[s])
        return points,okay

    targets=[];valids=[];point_errors=[];ray_errors=[];sens=[];sens_valid=[];invalid_frac=[];xyz=[]
    for used in ([0,1],[0,2],[0,1,2]):
        point,good=solve(used,observations)
        z=np.where(good,point[:,2],np.nan);target=rr*z[:,None]
        pe=np.full((n,3),np.nan);re=pe.copy()
        for s in used:
            for x,out in ((point,pe),(target,re)):
                projected=(x@ps[s][:,:3].T+ps[s][:,3])@cameras[s].T
                with np.errstate(invalid='ignore',divide='ignore'):
                    pix=projected[:,:2]/projected[:,2:]
                out[:,s]=np.where(good,np.linalg.norm(pix-observations[s],axis=-1),np.nan)
        deltas=[];perturbed_valid=[]
        for s in used[1:]:
            for axis in range(2):
                for sign in (-1.,1.):
                    changed=[q.copy() for q in observations];changed[s][:,axis]+=sign
                    new,new_good=solve(used,changed)
                    perturbed_valid.append(new_good&good)
                    deltas.append(np.abs(new[:,2]-z))
        pv=np.stack(perturbed_valid,axis=1);all_good=pv.all(1)
        delta=np.full(n,np.nan);delta[all_good]=np.stack(deltas,1)[all_good].max(1)
        targets.append(z);valids.append(good);point_errors.append(pe);ray_errors.append(re)
        sens.append(delta);sens_valid.append(all_good);invalid_frac.append(1-pv.mean(1));xyz.append(point)
    angle=[]
    a=rr/np.linalg.norm(rr,axis=1,keepdims=True)
    for i in range(2):
        r=ps[i+1][:,:3];b=rays(qs[:,i],src_ks[i])@r
        b/=np.linalg.norm(b,axis=1,keepdims=True)
        value=np.rad2deg(np.arccos(np.clip((a*b).sum(1),-1,1)))
        value[~forward_valid[:,i]]=np.nan;angle.append(value)
    depths=np.stack(targets,1)
    return dict(depth=depths,valid=np.stack(valids,1),points=np.stack(xyz,1),
        point_reprojection_px=np.stack(point_errors,1),ray_reprojection_px=np.stack(ray_errors,1),
        sensitivity_max_z_m=np.stack(sens,1),sensitivity_all_valid=np.stack(sens_valid,1),
        sensitivity_invalid_fraction=np.stack(invalid_frac,1),ray_angle_deg=np.stack(angle,1),
        pair_depth_disagreement_m=np.abs(depths[:,0]-depths[:,1]))


def pair_geometry(xy, q, forward_valid, ref_k, ref_tr, src_k, src_tr,
                  src_shape, ref_near, src_near):
    """Observed-pair conditioning, using the same normalized COLMAP solver.

    Depth is diagnostic only. Effective angle folds theta and 180-theta so
    antiparallel collinear rays cannot masquerade as excellent parallax.
    """
    xy=np.asarray(xy,float);q=np.asarray(q,float);n=len(xy)
    if xy.shape!=(n,2) or q.shape!=xy.shape:raise ValueError('invalid pair coordinates')
    r,t=relative_pose(ref_tr,src_tr);a=rays(xy,ref_k)
    projections=[np.c_[np.eye(3),np.zeros(3)],np.c_[r,t]]
    base=np.asarray(forward_valid,bool)&np.isfinite(xy).all(1)
    def solve(other):
        use=base&inside(other,src_shape);ids=np.flatnonzero(use)
        point=np.full((n,3),np.nan)
        if len(ids):
            value,good=_linear_points([a[ids,:2],rays(other[ids],src_k)[:,:2]],projections)
            point[ids]=value;use[ids]&=good
        source=point@r.T+t
        use&=np.isfinite(point).all(1)&(point[:,2]>ref_near)&(source[:,2]>src_near)
        return point,use
    point,valid=solve(q);depth=np.where(valid,point[:,2],np.nan)
    b=rays(q,src_k)@r
    with np.errstate(invalid='ignore',divide='ignore'):
        cos=(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        raw_angle=np.rad2deg(np.arccos(np.clip(cos,-1,1)))
    angle=np.minimum(raw_angle,180-raw_angle);angle[~valid]=np.nan
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    f=np.linalg.inv(src_k).T@tx@r@np.linalg.inv(ref_k)
    x=np.c_[xy,np.ones(n)];y=np.c_[q,np.ones(n)]
    l2=x@f.T;l1=y@f
    den=np.sqrt((l2[:,:2]**2).sum(1)+(l1[:,:2]**2).sum(1))
    epipolar=np.full(n,np.nan);good=base&np.isfinite(den)&(den>1e-15)
    epipolar[good]=np.abs((y*l2).sum(1)[good])/den[good]
    deltas=[];perturbed=[]
    for axis in range(2):
        for sign in (-1.,1.):
            other=q.copy();other[:,axis]+=sign
            p,pv=solve(other);perturbed.append(pv&valid);deltas.append(np.abs(p[:,2]-depth))
    pv=np.stack(perturbed,1);sv=pv.all(1);sensitivity=np.full(n,np.nan)
    sensitivity[sv]=np.stack(deltas,1)[sv].max(1)
    return dict(depth=depth,valid=valid,angle_deg=angle,raw_angle_deg=raw_angle,
        epipolar_px=epipolar,sensitivity_max_z_m=sensitivity,sensitivity_valid=sv,
        sensitivity_invalid_fraction=1-pv.mean(1))


def routing_evidence(costs, projection_valid, evidence, seed=0,
                     cycle_max_px=1., overlap_min=.5):
    """Freeze old/new shared supports, candidate ranks and source-quota control.

    Winners are offline labels only: -1 abstains, 0=current,1=DA3,2=MVS,3=spread.
    Candidate depths/costs never influence selection of observed supports.
    """
    costs=np.asarray(costs,float);valid=np.asarray(projection_valid,bool)&np.isfinite(costs)
    if costs.ndim!=3 or costs.shape[1]!=4 or valid.shape!=costs.shape or costs.shape[2]<2:
        raise ValueError('expected N x 4 x S with at least two supports')
    n,_,s=costs.shape
    for key in ('forward_valid','cycle_valid','cycle_px','overlap','pair_valid','angle_deg'):
        if np.asarray(evidence[key]).shape!=(n,s):raise ValueError('invalid support evidence: '+key)
    angle=np.asarray(evidence['angle_deg'])
    eligible=(evidence['forward_valid']&evidence['cycle_valid']&evidence['pair_valid']&
        np.isfinite(evidence['cycle_px'])&(evidence['cycle_px']<=cycle_max_px)&
        np.isfinite(evidence['overlap'])&(evidence['overlap']>=overlap_min)&np.isfinite(angle))
    order=np.argsort(np.where(eligible,-angle,np.inf),axis=1,kind='stable')[:,:2]
    old=np.broadcast_to(np.arange(2),(n,2)).copy()
    out={'support_eligible':eligible}
    for protocol,ix in (('old_first2',old),('new_best2',order)):
        good=np.take_along_axis(eligible,ix,axis=1)
        indices=np.where(good,ix,-1);mask=np.zeros((n,s),bool)
        rows=np.broadcast_to(np.arange(n)[:,None],(n,2))
        mask[rows[good],ix[good]]=True
        out[protocol+'_support_indices']=indices;out[protocol+'_support_mask']=mask
        for count in (3,4):
            p=f'{protocol}_common{count}'
            domain=(mask.sum(1)==2)&np.all(valid[:,:count]|~mask[:,None,:],axis=(1,2))
            means=np.full((n,count),np.nan)
            means[domain]=np.where(mask[:,None,:],costs[:,:count],0).sum(2)[domain]/2
            winner=np.full(n,-1,np.int8);margin=np.full(n,np.nan)
            if domain.any():
                m=means[domain];best=np.argmin(m,axis=1)
                unique=(m==m.min(1)[:,None]).sum(1)==1
                winner[domain]=np.where(unique,best,-1)
                ordered=np.sort(m,axis=1);margin[domain]=ordered[:,1]-ordered[:,0]
            agreement=np.zeros(n,np.int8)
            for j in range(2):
                chosen=np.take_along_axis(costs[:,:count],ix[:,None,j:j+1],axis=2)[:,:,0]
                vote=chosen.argmin(1);unique=(chosen==chosen.min(1)[:,None]).sum(1)==1
                agreement+=((winner>=0)&(vote==winner)&unique).astype(np.int8)
            random=winner.copy();ids=np.flatnonzero(domain)
            random[ids]=np.random.default_rng(int(seed)).permutation(winner[ids])
            out.update({p+'_domain':domain,p+'_means':means,p+'_winner':winner,
                p+'_margin_px':margin,p+'_support_agreement_count':agreement,p+'_random_winner':random})
    return out
