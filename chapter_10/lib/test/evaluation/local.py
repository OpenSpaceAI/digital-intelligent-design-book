from lib.test.evaluation.environment import EnvSettings
import os
prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = prj_dir
    settings.save_dir = os.path.join(prj_dir, 'output')
    settings.result_plot_path = os.path.join(prj_dir, 'test/result_plots')
    settings.results_path = os.path.join(prj_dir, 'test/tracking_results')
    settings.lasot_path = os.path.join(prj_dir, 'data/lasot')
    settings.nfs_path = os.path.join(prj_dir, 'data/nfs')
    settings.otb_path = os.path.join(prj_dir, 'data/otb99')
    settings.trackingnet_path = os.path.join(prj_dir, 'data/trackingnet')
    settings.uav_path = os.path.join(prj_dir, 'data/uav')
    settings.tnl2k_path = os.path.join(prj_dir, 'data/tnl2k/test')
    settings.otb99_path = os.path.join(prj_dir, 'data/otb99')
    settings.lasot_ext_path = os.path.join(prj_dir, 'data/lasotext')
    settings.got10k_path = os.path.join(prj_dir, 'data/got10k')

    return settings
    # settings.davis_dir = ''
    # settings.got10k_lmdb_path = '/ssd/wqj/project/ustc_moting/data/got10k_lmdb'
    # settings.got10k_path = '/ssd/wqj/project/ustc_moting/data/got10k'
    # settings.got_packed_results_path = ''
    # settings.got_reports_path = ''
    # settings.lasot_lmdb_path = '/ssd/wqj/project/ustc_moting/data/lasot_lmdb'
    # settings.lasot_path = '/ssd/wqj/project/ustc_moting/data/lasot'
    # settings.network_path = '/ssd/tyy/projects/OSTrack/test/networks'    # Where tracking networks are stored.
    # settings.nfs_path = '/data1/dataset/VOT/NFS'
    # settings.otb_path = '/ssd/wqj/project/ustc_moting/data/OTB2015'
    # settings.prj_dir = '/ssd/tyy/projects/OSTrack'
    # settings.result_plot_path = '/ssd/tyy/projects/OSTrack/test/result_plots'
    # settings.results_path = '/ssd/tyy/projects/OSTrack/test/tracking_results'    # Where to store tracking results
    # settings.save_dir = '/ssd/tyy/projects/OSTrack/output'
    # settings.segmentation_path = '/ssd/tyy/projects/OSTrack/test/segmentation_results'
    # settings.tc128_path = '/ssd/wqj/project/ustc_moting/data/TC128'
    # settings.tn_packed_results_path = ''
    # settings.tpl_path = ''
    # settings.trackingnet_path = '/ssd/wqj/project/ustc_moting/data/trackingnet'
    # settings.uav_path = '/data1/dataset/VOT/UAV123/UAV123'
    # settings.vot_path = '/ssd/wqj/project/ustc_moting/data/VOT2019'
    # settings.youtubevos_dir = ''
    # settings.tnl2k_path = '/ssd/dataset/VOT/TNL2K/test'
    # settings.otb99_path = '/ssd/dataset/VOT/OTB-lang/OTB_sentences'
    # settings.lasot_ext_path = '/ssd/dataset/MOT/LaSOT_extension_subset'

    #return settings