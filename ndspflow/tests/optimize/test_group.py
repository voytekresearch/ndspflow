"""Test group refitting functions."""

from ndspflow.optimize.group import refit_group


def test_refit_group(fooof_outs, test_data):

    fg = fooof_outs['fg']
    sigs = test_data['sig_2d']
    fs = test_data['fs']
    f_range = test_data['f_range']

    fg_refit, imfs, pe_masks = refit_group(fg, sigs, fs, f_range, refit_ap=True)

    for imf in imfs[1:]:
        assert (imfs[0] == imf).all()

    for mask in pe_masks[1:]:
        assert (pe_masks[0] == mask).all()

    fm_ref = fg_refit.get_fooof(0)
    for fm_idx in range(1, len(fg_refit)):
        fm = fg_refit.get_fooof(fm_idx)
        assert (fm._peak_fit == fm_ref._peak_fit).all()
        assert (fm._ap_fit == fm_ref._ap_fit).all()

    fm_ref_orig = fg.get_fooof(0)
    assert not (fm_ref_orig._peak_fit == fm_ref._peak_fit).all()
    assert not (fm_ref_orig._ap_fit == fm_ref._ap_fit).all()
