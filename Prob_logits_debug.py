def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
    # ... your existing code ...
    
    # After computing prob_logits and target_m_index, add:
    with torch.no_grad():
        probs = F.softmax(prob_logits, dim=-1)  # (bs, Ka)
        
        # How peaked are the predictions?
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        max_prob = probs.max(dim=-1).values.mean()
        
        # Are all 24 modes being used or just a few?
        pred_modes = prob_logits.argmax(dim=-1)  # (bs,)
        num_unique = torch.unique(pred_modes).numel()
        
        # How often does the top predicted mode match WTA winner?
        top1_acc = (pred_modes == target_m_index).float().mean()
        
        self.log("cls/prob_entropy", entropy, on_step=True, on_epoch=True)
        self.log("cls/max_prob", max_prob, on_step=True, on_epoch=True)
        self.log("cls/num_unique_modes_used", float(num_unique), on_step=True, on_epoch=True)
        self.log("cls/top1_accuracy", top1_acc, on_step=True, on_epoch=True)
        
        # WTA stability: log the distribution of winning mode indices
        wta_counts = torch.bincount(target_m_index, minlength=24).float()
        wta_entropy = -(wta_counts/wta_counts.sum() * 
                       torch.log(wta_counts/wta_counts.sum() + 1e-9)).sum()
        self.log("cls/wta_target_entropy", wta_target_entropy, on_step=True, on_epoch=True)
    
    cls_loss = F.cross_entropy(prob_logits, target_m_index.detach())
    return reg_loss, cls_loss, collision_loss
